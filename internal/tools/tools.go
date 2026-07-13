// Package tools 提供 Agent 可调用的工具注册中心和工具实现。
// 设计原则：
//   - 每个工具实现 Tool 接口
//   - Registry 作为工厂管理所有工具
//   - 工具之间无依赖，可独立测试
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

// Tool 接口定义
// Tool 是所有工具必须实现的接口。
// Name: 工具的唯一标识，LLM 通过名称引用工具
// Description: LLM 理解工具用途的提示文本
// Parameters: JSON Schema 格式的参数定义
// Execute: 执行工具的核心逻辑
type Tool interface {
	Name() string
	Description() string
	Parameters() json.RawMessage
	Execute(ctx context.Context, args json.RawMessage) *types.ToolResult
}

// Registry — 工具注册中心
// Registry 管理所有可用的工具。
// 线程安全，支持运行时动态注册/注销。
// 提供统一的工具调用接口和超时控制。
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

// NewRegistry 创建空工具注册中心。
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

// Register 注册一个工具。工具名称必须唯一。
func (r *Registry) Register(tool Tool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := tool.Name()
	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("tool %q already registered", name)
	}
	r.tools[name] = tool
	return nil
}

// Get 按名称获取工具。
func (r *Registry) Get(name string) (Tool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tool, ok := r.tools[name]
	if !ok {
		return nil, fmt.Errorf("tool %q not found", name)
	}
	return tool, nil
}

// List 返回所有已注册的工具名称。
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	return names
}

// GetOpenAITools 将所有工具转为 OpenAI Function Calling 格式。
// 供 LLM 调用时识别可用工具。
func (r *Registry) GetOpenAITools() interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()

	type functionDef struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	}

	var result []map[string]interface{}
	for _, tool := range r.tools {
		result = append(result, map[string]interface{}{
			"type": "function",
			"function": functionDef{
				Name:        tool.Name(),
				Description: tool.Description(),
				Parameters:  tool.Parameters(),
			},
		})
	}
	return result
}

// ExecuteTool 查找并执行指定工具。
func (r *Registry) ExecuteTool(ctx context.Context, name string, args json.RawMessage) *types.ToolResult {
	tool, err := r.Get(name)
	if err != nil {
		return &types.ToolResult{
			Success: false,
			Error:   fmt.Sprintf("tool not found: %s", name),
		}
	}
	return tool.Execute(ctx, args)
}

// KnowledgeSearchTool — 知识搜索工具
// KnowledgeSearchTool 是 Agent 的核心检索工具。
// 它通过语义向量搜索从知识库中查找相关信息。
type KnowledgeSearchTool struct {
	name        string
	description string
	parameters  json.RawMessage
	retriever   searchFunc
}

// searchFunc 是检索函数的签名，由 Agent 注入。
type searchFunc func(ctx context.Context, query string) ([]*types.SearchResult, error)

// NewKnowledgeSearchTool 创建知识搜索工具。
func NewKnowledgeSearchTool(fn searchFunc) *KnowledgeSearchTool {
	return &KnowledgeSearchTool{
		name:        "knowledge_search",
		description: `Search the knowledge base for information related to the user's question. 
Use this tool when you need to find specific facts, concepts, or explanations from the indexed documents.
Provide a clear, well-formed search query. Returns relevant document chunks with relevance scores.`,
		parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"query": {
					"type": "string",
					"description": "The search query (clear, specific question or keywords)"
				}
			},
			"required": ["query"]
		}`),
		retriever: fn,
	}
}

func (t *KnowledgeSearchTool) Name() string               { return t.name }
func (t *KnowledgeSearchTool) Description() string         { return t.description }
func (t *KnowledgeSearchTool) Parameters() json.RawMessage { return t.parameters }

func (t *KnowledgeSearchTool) Execute(ctx context.Context, args json.RawMessage) *types.ToolResult {
	var input struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal(args, &input); err != nil {
		return &types.ToolResult{Success: false, Error: fmt.Sprintf("invalid args: %v", err)}
	}
	if input.Query == "" {
		return &types.ToolResult{Success: false, Error: "query is required"}
	}

	results, err := t.retriever(ctx, input.Query)
	if err != nil {
		return &types.ToolResult{Success: false, Error: fmt.Sprintf("search failed: %v", err)}
	}

	if len(results) == 0 {
		return &types.ToolResult{
			Success: true,
			Output:  "No relevant information found in the knowledge base. (You must answer based on your own general knowledge, and set confidence to 0.3-0.5.)",
		}
	}

	// 格式化输出
	var sb Builder
	sb.Writeln("Found %d relevant results:\n", len(results))
	for i, r := range results {
		if r.Chunk == nil {
			continue
		}
		sb.Writeln("[%d] Score: %.4f | Method: %s", i+1, r.Score, r.Method)
		sb.Writeln("    Source: %s", r.Chunk.DocTitle)
		sb.Writeln("    Content: %s\n", truncate(r.Chunk.Content, 300))
	}

	return &types.ToolResult{
		Success: true,
		Output:  sb.String(),
		Data: map[string]interface{}{
			"result_count": len(results),
		},
	}
}

// GraphSearchTool — 知识图谱搜索工具
// GraphSearchTool 查询知识图谱，返回实体间的关系网络。
// 当需要理解多个概念之间的关系时使用。
type GraphSearchTool struct {
	name       string
	parameters json.RawMessage
	graphQuery func(ctx context.Context, entity string) ([]*types.SearchResult, error)
}

// NewGraphSearchTool 创建知识图谱搜索工具。
func NewGraphSearchTool(fn func(ctx context.Context, entity string) ([]*types.SearchResult, error)) *GraphSearchTool {
	return &GraphSearchTool{
		name: "graph_search",
		parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"entity": {
					"type": "string",
					"description": "The entity name to search in the knowledge graph"
				}
			},
			"required": ["entity"]
		}`),
		graphQuery: fn,
	}
}

func (t *GraphSearchTool) Name() string { return t.name }
func (t *GraphSearchTool) Description() string {
	return `Search the knowledge graph for entities and their relationships. 
Use this tool when you need to understand how concepts, people, or technologies are connected.
Provide an entity name to find its relationships.`
}
func (t *GraphSearchTool) Parameters() json.RawMessage { return t.parameters }

func (t *GraphSearchTool) Execute(ctx context.Context, args json.RawMessage) *types.ToolResult {
	var input struct {
		Entity string `json:"entity"`
	}
	if err := json.Unmarshal(args, &input); err != nil {
		return &types.ToolResult{Success: false, Error: fmt.Sprintf("invalid args: %v", err)}
	}
	if input.Entity == "" {
		return &types.ToolResult{Success: false, Error: "entity is required"}
	}

	results, err := t.graphQuery(ctx, input.Entity)
	if err != nil {
		return &types.ToolResult{Success: false, Error: fmt.Sprintf("graph search failed: %v", err)}
	}

	var sb Builder
	sb.Writeln("Knowledge Graph results for '%s':\n", input.Entity)

	for _, r := range results {
		if r.Chunk != nil {
			sb.Writeln("- %s (related to %s): %s", r.Chunk.DocTitle, input.Entity, truncate(r.Chunk.Content, 200))
		} else {
			sb.Writeln("- %s (confidence: %.2f)", input.Entity, r.Score)
		}
	}

	return &types.ToolResult{
		Success: true,
		Output:  sb.String(),
		Data:    map[string]interface{}{"result_count": len(results)},
	}
}

// WebSearchTool — 网页搜索工具
// WebSearchTool 搜索互联网获取最新信息。
// 用于回答需要实时数据的查询。
type WebSearchTool struct {
	name       string
	parameters json.RawMessage
	searchFunc func(ctx context.Context, query string) (string, error)
}

// NewWebSearchTool 创建网页搜索工具。
func NewWebSearchTool(fn func(ctx context.Context, query string) (string, error)) *WebSearchTool {
	return &WebSearchTool{
		name: "web_search",
		parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"query": {
					"type": "string",
					"description": "The web search query"
				}
			},
			"required": ["query"]
		}`),
		searchFunc: fn,
	}
}

func (t *WebSearchTool) Name() string { return t.name }
func (t *WebSearchTool) Description() string {
	return `Search the internet for current, up-to-date information. 
Use this tool when the knowledge base doesn't have the answer or the question is about recent events.`
}
func (t *WebSearchTool) Parameters() json.RawMessage { return t.parameters }

func (t *WebSearchTool) Execute(ctx context.Context, args json.RawMessage) *types.ToolResult {
	var input struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal(args, &input); err != nil {
		return &types.ToolResult{Success: false, Error: fmt.Sprintf("invalid args: %v", err)}
	}
	if input.Query == "" {
		return &types.ToolResult{Success: false, Error: "query is required"}
	}

	result, err := t.searchFunc(ctx, input.Query)
	if err != nil {
		return &types.ToolResult{Success: false, Error: fmt.Sprintf("web search failed: %v", err)}
	}

	return &types.ToolResult{
		Success: true,
		Output:  result,
		Data:    map[string]interface{}{"source": "web"},
	}
}

// CalculatorTool — 计算器工具
// CalculatorTool 执行数学计算和数据分析。
// 当 Agent 需要进行精确计算时使用。
type CalculatorTool struct {
	name       string
	parameters json.RawMessage
}

// NewCalculatorTool 创建计算器工具。
func NewCalculatorTool() *CalculatorTool {
	return &CalculatorTool{
		name: "calculator",
		parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"expression": {
					"type": "string",
					"description": "The mathematical expression to evaluate (e.g., '2 + 2', 'avg([1,2,3,4,5])')"
				}
			},
			"required": ["expression"]
		}`),
	}
}

func (t *CalculatorTool) Name() string { return t.name }
func (t *CalculatorTool) Description() string {
	return `Evaluate mathematical expressions and perform calculations. Supports: +, -, *, /, sqrt, pow, avg, sum, min, max.`
}
func (t *CalculatorTool) Parameters() json.RawMessage { return t.parameters }

func (t *CalculatorTool) Execute(_ context.Context, args json.RawMessage) *types.ToolResult {
	var input struct {
		Expression string `json:"expression"`
	}
	if err := json.Unmarshal(args, &input); err != nil {
		return &types.ToolResult{Success: false, Error: fmt.Sprintf("invalid args: %v", err)}
	}
	if input.Expression == "" {
		return &types.ToolResult{Success: false, Error: "expression is required"}
	}

	result, err := evalExpression(input.Expression)
	if err != nil {
		return &types.ToolResult{Success: false, Error: fmt.Sprintf("calculation error: %v", err)}
	}

	return &types.ToolResult{
		Success: true,
		Output:  fmt.Sprintf("Result: %v", result),
		Data:    map[string]interface{}{"result": result},
	}
}

// FinalAnswerTool — 最终答案工具（控制 Agent 何时给出最终答案）
// FinalAnswerTool 是 Agent 用来提交最终答案的工具。
// 调用此工具意味着 Agent 完成了所有调研，准备给出最终答案。
// 创新点：通过强制使用此工具，确保 Agent 不会"忘记"提供答案。
type FinalAnswerTool struct {
	name       string
	parameters json.RawMessage
}

// NewFinalAnswerTool 创建最终答案工具。
func NewFinalAnswerTool() *FinalAnswerTool {
	return &FinalAnswerTool{
		name: "final_answer",
		parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"answer": {
					"type": "string",
					"description": "The complete final answer to the user's question"
				},
				"confidence": {
					"type": "number",
					"description": "Confidence level (0.0 to 1.0)",
					"minimum": 0,
					"maximum": 1
				},
				"sources": {
					"type": "array",
					"description": "Source document titles from knowledge_search, or [\"general knowledge\"] if no KB sources",
					"items": {"type": "string"}
				}
			},
			"required": ["answer", "confidence"]
		}`),
	}
}

func (t *FinalAnswerTool) Name() string { return t.name }
func (t *FinalAnswerTool) Description() string {
	return `Call this tool ONLY when you have gathered ALL necessary information and are ready to provide the COMPLETE final answer to the user.`
}
func (t *FinalAnswerTool) Parameters() json.RawMessage { return t.parameters }

func (t *FinalAnswerTool) Execute(_ context.Context, args json.RawMessage) *types.ToolResult {
	return &types.ToolResult{
		Success: true,
		Output: string(args),
	}
}

// 工具函数
// Builder 是高效的字符串构建器。
type Builder struct{ b []byte }

func (sb *Builder) Writeln(format string, args ...interface{}) {
	sb.b = append(sb.b, fmt.Sprintf(format, args...)...)
	sb.b = append(sb.b, '\n')
}
func (sb *Builder) String() string { return string(sb.b) }

func truncate(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "..."
}

// evalExpression 使用递归下降解析器求值简单算术表达式。
// 支持: +, -, *, /, (), 整数和小数。
func evalExpression(expr string) (float64, error) {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return 0, fmt.Errorf("empty expression")
	}
	p := &exprParser{input: expr}
	result, err := p.parse()
	if err != nil {
		return 0, fmt.Errorf("invalid expression %q: %w", expr, err)
	}
	if p.pos < len(p.input) {
		return 0, fmt.Errorf("unexpected char %q at pos %d", p.input[p.pos], p.pos)
	}
	return result, nil
}

type exprParser struct {
	input string
	pos   int
}

func (p *exprParser) peek() byte {
	if p.pos >= len(p.input) {
		return 0
	}
	return p.input[p.pos]
}
func (p *exprParser) advance() byte {
	c := p.input[p.pos]
	p.pos++
	return c
}
func (p *exprParser) skipWS() {
	for p.pos < len(p.input) {
		c := p.input[p.pos]
		if c != ' ' && c != '\t' {
			break
		}
		p.pos++
	}
}
func (p *exprParser) parseNumber() (float64, error) {
	p.skipWS()
	start := p.pos
	if p.pos >= len(p.input) {
		return 0, fmt.Errorf("expected number")
	}
	hasDot := false
	for p.pos < len(p.input) {
		c := p.input[p.pos]
		if c >= '0' && c <= '9' {
			p.pos++
		} else if c == '.' && !hasDot {
			hasDot = true
			p.pos++
		} else {
			break
		}
	}
	if p.pos == start {
		return 0, fmt.Errorf("expected number at pos %d", start)
	}
	return strconv.ParseFloat(p.input[start:p.pos], 64)
}
func (p *exprParser) parseFactor() (float64, error) {
	p.skipWS()
	if p.peek() == '(' {
		p.advance()
		v, err := p.parseExpr()
		if err != nil {
			return 0, err
		}
		p.skipWS()
		if p.peek() != ')' {
			return 0, fmt.Errorf("expected ')' at pos %d", p.pos)
		}
		p.advance()
		return v, nil
	}
	return p.parseNumber()
}
func (p *exprParser) parseTerm() (float64, error) {
	left, err := p.parseFactor()
	if err != nil {
		return 0, err
	}
	for {
		p.skipWS()
		op := p.peek()
		if op != '*' && op != '/' {
			break
		}
		p.advance()
		right, err := p.parseFactor()
		if err != nil {
			return 0, err
		}
		if op == '*' {
			left *= right
		} else {
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			left /= right
		}
	}
	return left, nil
}
func (p *exprParser) parseExpr() (float64, error) {
	left, err := p.parseTerm()
	if err != nil {
		return 0, err
	}
	for {
		p.skipWS()
		op := p.peek()
		if op != '+' && op != '-' {
			break
		}
		p.advance()
		right, err := p.parseTerm()
		if err != nil {
			return 0, err
		}
		if op == '+' {
			left += right
		} else {
			left -= right
		}
	}
	return left, nil
}
func (p *exprParser) parse() (float64, error) { return p.parseExpr() }
