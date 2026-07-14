// Package agent 提供 VolSeek-Agent 的核心引擎。
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sashabaranov/go-openai"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/llm"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/memory"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/rag"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/tools"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

type AgentEngine struct {
	config    *types.AgentConfig
	llm       *llm.Client
	registry  *tools.Registry
	router    *rag.QueryRouter
	retriever *rag.Retriever
	memory    memory.Memory
}

func NewAgentEngine(
	config *types.AgentConfig,
	llmClient *llm.Client,
	registry *tools.Registry,
	router *rag.QueryRouter,
	retriever *rag.Retriever,
	mem memory.Memory,
) *AgentEngine {
	return &AgentEngine{
		config:    config,
		llm:       llmClient,
		registry:  registry,
		router:    router,
		retriever: retriever,
		memory:    mem,
	}
}

func (e *AgentEngine) Execute(ctx context.Context, query string) (<-chan types.StreamEvent, error) {
	ch := make(chan types.StreamEvent, 128)
	go func() {
		defer close(ch)
		e.executeWithEvents(ctx, query, ch)
	}()
	return ch, nil
}

func (e *AgentEngine) executeWithEvents(ctx context.Context, query string, ch chan<- types.StreamEvent) {
	state := &types.AgentState{}
	e.sendEvent(ch, types.EventPlan, "Analyzing your question...", 0, 0, false)

	plan, err := e.createPlan(ctx, query)
	if err != nil {
		e.sendEvent(ch, types.EventError, fmt.Sprintf("Planning failed: %v", err), 0, 0, true)
		return
	}
	state.Plan = plan

	planContent := fmt.Sprintf("Plan: %s\nSub-goals:\n", plan.Goal)
	for i, sg := range plan.SubGoals {
		planContent += fmt.Sprintf("  %d. %s\n", i+1, sg)
	}
	e.sendEvent(ch, types.EventPlan, planContent, 0, len(plan.SubGoals), false)

	systemPrompt := e.buildSystemPrompt(ctx, query, plan)
	messages := []types.LLMMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: query},
	}
	openAITools := e.buildOpenAITools()

	stepIndex := 0
	maxRounds := e.config.MaxPlanningRounds
	if maxRounds <= 0 { maxRounds = 10 }

	for stepIndex < maxRounds {
		select {
		case <-ctx.Done():
			e.sendEvent(ch, types.EventError, "Cancelled", stepIndex, len(plan.SubGoals), true)
			return
		default:
		}
		e.sendEvent(ch, types.EventThink, fmt.Sprintf("Step %d: Thinking...", stepIndex+1), stepIndex, len(plan.SubGoals), false)

		resp, err := e.llm.ChatWithRetry(ctx, messages, openAITools, e.config.Temperature)
		if err != nil {
			e.sendEvent(ch, types.EventError, fmt.Sprintf("LLM call failed: %v", err), stepIndex, len(plan.SubGoals), true)
			return
		}

		step := types.Step{Index: stepIndex, Thought: resp.Content, Timestamp: time.Now()}
		needFinalAnswer := false

		if len(resp.ToolCalls) > 0 {
			for _, tc := range resp.ToolCalls {
				if tc.Name == "final_answer" {
					needFinalAnswer = true
					var args struct {
						Answer     string   `json:"answer"`
						Confidence float64  `json:"confidence"`
						Sources    []string `json:"sources"`
					}
					if err := json.Unmarshal([]byte(tc.Arguments), &args); err == nil {
						state.FinalAnswer = args.Answer
						state.Confidence = args.Confidence
						for _, s := range args.Sources {
							state.Sources = append(state.Sources, types.SourceRef{Title: s})
						}
					}
					break
				}
			}
			if !needFinalAnswer {
				step.ToolCalls = e.executeToolCalls(ctx, resp.ToolCalls, ch, stepIndex)
				step.Observation = e.formatObservations(step.ToolCalls)
				messages = append(messages, types.LLMMessage{Role: "assistant", Content: resp.Content, ToolCalls: resp.ToolCalls})
				for _, tc := range step.ToolCalls {
					output := tc.Result.Output
					if !tc.Result.Success { output = fmt.Sprintf("Error: %s", tc.Result.Error) }
					messages = append(messages, types.LLMMessage{Role: "tool", ToolCallID: tc.ID, Content: output})
				}
				state.Steps = append(state.Steps, step)
				stepIndex++
				continue
			}
		} else if resp.FinishReason == "stop" {
			if resp.Content != "" {
				state.FinalAnswer = resp.Content; state.Confidence = 0.7; needFinalAnswer = true
			} else {
				state.FinalAnswer = "Unable to generate a response."; state.Confidence = 0.0; needFinalAnswer = true
			}
		}
		if needFinalAnswer { break }
		stepIndex++
	}

	if e.config.EnableReflection && state.FinalAnswer != "" {
		e.sendEvent(ch, types.EventReflect, "Reviewing answer quality...", stepIndex, len(plan.SubGoals), false)
		refinedAnswer, confidence, err := e.reflectOnAnswer(ctx, query, state.FinalAnswer)
		if err == nil { state.FinalAnswer = refinedAnswer; state.Confidence = confidence }
	}

	state.IsComplete = true
	if state.FinalAnswer == "" {
		state.FinalAnswer = "Please try rephrasing your question."; state.Confidence = 0.0
	}

	// 将最终答案按小块逐块发送，实现流式输出效果
	chunks := splitIntoChunks(state.FinalAnswer, 3)
	for _, chunk := range chunks {
		e.sendEvent(ch, types.EventAnswer, chunk, stepIndex, len(plan.SubGoals), false)
		time.Sleep(15 * time.Millisecond)
	}
	ch <- types.StreamEvent{Type: types.EventDone, Content: state.FinalAnswer,
		Data: map[string]interface{}{"confidence": state.Confidence, "steps": len(state.Steps), "sources": state.Sources}, Done: true}

	// 保存到记忆
	if e.memory != nil {
		sid := e.sessionID(ctx)
		e.memory.Save(ctx, sid, memory.Entry{Role: "user", Content: query, SessionID: sid, Timestamp: time.Now()})
		e.memory.Save(ctx, sid, memory.Entry{Role: "assistant", Content: state.FinalAnswer, SessionID: sid, Timestamp: time.Now()})
	}
}

// buildSystemPrompt 构建提示词，注入记忆上下文。
func (e *AgentEngine) buildSystemPrompt(ctx context.Context, query string, plan *types.Plan) string {
	toolNames := e.registry.List()
	memCtx := e.getMemoryContext(ctx)

	return fmt.Sprintf(`You are VolSeek-Agent, an intelligent research assistant.

## Conversation History
%s

## Plan
Goal: %s
Steps: %s

## Available Tools
%s

## Rules
1. Use tools when you need information.
2. Call "final_answer" with your complete answer.
3. Respond in the same language as the user.`,
		memCtx, plan.Goal, strings.Join(plan.SubGoals, " -> "), strings.Join(toolNames, ", "))
}

func (e *AgentEngine) getMemoryContext(ctx context.Context) string {
	if e.memory == nil { return "(no previous conversation)" }
	sid := e.sessionID(ctx)
	entries, err := e.memory.Recall(ctx, sid, 10)
	if err != nil || len(entries) == 0 { return "(no previous conversation)" }
	var sb strings.Builder
	for _, entry := range entries {
		role := entry.Role
		if role == "assistant" { role = "you" }
		sb.WriteString(fmt.Sprintf("[%s] %s\n", role, entry.Content))
	}
	return sb.String()
}

// sessionID 从 context 中提取会话 ID，未设置时返回 "default"。
func (e *AgentEngine) sessionID(ctx context.Context) string {
	if sid, ok := ctx.Value("session_id").(string); ok && sid != "" {
		return sid
	}
	return "default"
}

// (remaining functions unchanged: createPlan, parsePlanJSON, executeToolCalls, etc.)
// Keep all the original helper methods below.


func (e *AgentEngine) createPlan(ctx context.Context, query string) (*types.Plan, error) {
	// 先用意图路由器分析查询
	intent, err := e.router.Analyze(ctx, query)
	if err != nil {
		intent = &types.QueryIntent{Type: types.QueryConceptual, Summary: query}
	}

	// 获取可用工具列表
	toolNames := e.registry.List()
	toolsList := strings.Join(toolNames, ", ")

	// 让 LLM 生成计划
	planPrompt := fmt.Sprintf(`You are a planning agent. Analyze the user's question and create a structured execution plan.

Available tools: %s

Respond with ONLY a JSON object:
{
  "goal": "Restate the user's goal clearly",
  "sub_goals": ["Step 1 description", "Step 2 description", ...],
  "requires_tool": true/false,
  "reasoning": "Why this plan is appropriate"
}`, toolsList)

	messages := []types.LLMMessage{
		{Role: "system", Content: planPrompt},
		{Role: "user", Content: query},
	}

	resp, err := e.llm.Chat(ctx, messages, nil, 0.3) // 低温度确保计划稳定
	if err != nil {
		// LLM 失败时，创建一个默认计划
		return &types.Plan{
			Goal:        query,
			SubGoals:    []string{"Search for relevant information", "Synthesize the answer"},
			RequiresTool: true,
			Reasoning:   "Default plan (LLM planning failed)",
		}, nil
	}

	plan := e.parsePlanJSON(resp.Content)
	if plan == nil {
		// JSON 解析失败，使用默认计划
		return &types.Plan{
			Goal:         query,
			SubGoals:     []string{"Understand the question", "Find relevant information", "Formulate answer"},
			RequiresTool: intent.Type != types.QueryRecent, // 实时查询需要 web search
			Reasoning:    intent.Summary,
		}, nil
	}

	return plan, nil
}

// parsePlanJSON 解析 LLM 返回的 JSON 格式计划。
func (e *AgentEngine) parsePlanJSON(jsonStr string) *types.Plan {
	// 提取 JSON 部分
	if idx := strings.Index(jsonStr, "{"); idx >= 0 {
		if end := strings.LastIndex(jsonStr, "}"); end > idx {
			jsonStr = jsonStr[idx : end+1]
		}
	}

	var plan types.Plan
	if err := json.Unmarshal([]byte(jsonStr), &plan); err != nil {
		return nil
	}
	if plan.Goal == "" {
		return nil
	}
	return &plan
}

// 阶段 2: 工具执行 (Tool Execution)
// executeToolCalls 执行 LLM 请求的所有工具调用。
// 支持并行执行：当多个工具独立时，并发调用以提高效率。
func (e *AgentEngine) executeToolCalls(
	ctx context.Context,
	calls []types.ToolCallDef,
	ch chan<- types.StreamEvent,
	stepIndex int,
) []types.ToolCall {
	if len(calls) == 0 {
		return nil
	}

	results := make([]types.ToolCall, len(calls))

	// 是否并行执行
	if e.config.ParallelToolCalls && len(calls) > 1 {
		e.executeParallel(ctx, calls, results, ch, stepIndex)
	} else {
		for i, tc := range calls {
			results[i] = e.executeSingle(ctx, tc, ch, stepIndex)
		}
	}

	return results
}

// executeSingle 执行单个工具调用。
func (e *AgentEngine) executeSingle(ctx context.Context, tc types.ToolCallDef, ch chan<- types.StreamEvent, stepIndex int) types.ToolCall {
	e.sendEvent(ch, types.EventToolCall, fmt.Sprintf("Calling tool: %s", tc.Name), stepIndex, 0, false)

	start := time.Now()
	result := e.registry.ExecuteTool(ctx, tc.Name, json.RawMessage(tc.Arguments))
	duration := time.Since(start)

	toolCall := types.ToolCall{
		ID:       tc.ID,
		Name:     tc.Name,
		Args:     json.RawMessage(tc.Arguments),
		Result:   result,
		Duration: duration,
	}

	status := "✅"
	if !result.Success {
		status = "❌"
	}

	outputPreview := truncate(result.Output, 200)
	e.sendEvent(ch, types.EventToolResult,
		fmt.Sprintf("%s Tool '%s' completed in %v\n%s", status, tc.Name, duration.Round(time.Millisecond), outputPreview),
		stepIndex, 0, false)

	return toolCall
}

// executeParallel 并行执行多个独立工具调用。
// 使用 sync.WaitGroup 而非 errgroup，确保一个工具失败不会取消其他工具。
func (e *AgentEngine) executeParallel(
	ctx context.Context,
	calls []types.ToolCallDef,
	results []types.ToolCall,
	ch chan<- types.StreamEvent,
	stepIndex int,
) {
	e.sendEvent(ch, types.EventToolCall,
		fmt.Sprintf("Executing %d tools in parallel...", len(calls)),
		stepIndex, 0, false)

	var mu sync.Mutex
	var wg sync.WaitGroup

	for i, tc := range calls {
		i, tc := i, tc
		wg.Add(1)
		go func() {
			defer wg.Done()
			tcResult := e.executeSingle(ctx, tc, ch, stepIndex)
			mu.Lock()
			results[i] = tcResult
			mu.Unlock()
		}()
	}

	wg.Wait()
}

// 阶段 3: 反思 (Self-Reflection)
// reflectOnAnswer 让 Agent 对自己生成的答案进行质量评审。
// 这是区别于传统 RAG 的关键创新：Agent 不仅回答问题，还检查自己的回答。
// 评审维度：
//   - 准确性：答案是否有事实错误
//   - 完整性：是否回答了用户的所有问题点
//   - 来源引用：是否清楚地标明了信息来源
//   - 清晰度：表达是否清晰易懂
func (e *AgentEngine) reflectOnAnswer(ctx context.Context, query, answer string) (string, float64, error) {
	reflectPrompt := fmt.Sprintf(`You are a quality reviewer. Review the following Q&A pair and provide feedback.

USER QUESTION: %s

DRAFT ANSWER: %s

Evaluate the answer on:
1. Accuracy (0-1): Are there any factual errors?
2. Completeness (0-1): Does it fully answer the question?
3. Clarity (0-1): Is it well-structured and easy to understand?
4. Source attribution: Does it cite sources?

Respond with ONLY a JSON object:
{
  "refined_answer": "The improved answer (fix any issues found)",
  "confidence": 0.0-1.0,
  "issues_found": ["issue1", "issue2"],
  "improvements": ["improvement1", "improvement2"]
}`, query, answer)

	messages := []types.LLMMessage{
		{Role: "system", Content: "You are a thorough but fair answer reviewer."},
		{Role: "user", Content: reflectPrompt},
	}

	resp, err := e.llm.Chat(ctx, messages, nil, 0.2) // 低温度确保评审一致性
	if err != nil {
		return answer, 0.7, err
	}

	// 解析评审结果
	type reflectResult struct {
		RefinedAnswer string   `json:"refined_answer"`
		Confidence    float64  `json:"confidence"`
		IssuesFound   []string `json:"issues_found"`
	}

	var result reflectResult
	jsonStr := resp.Content
	if idx := strings.Index(jsonStr, "{"); idx >= 0 {
		if end := strings.LastIndex(jsonStr, "}"); end > idx {
			jsonStr = jsonStr[idx : end+1]
		}
	}

	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return answer, 0.7, nil
	}

	if result.RefinedAnswer != "" {
		return result.RefinedAnswer, result.Confidence, nil
	}

	return answer, 0.7, nil
}

// 辅助方法
// buildOpenAITools 将注册的工具转为 OpenAI 格式。
func (e *AgentEngine) buildOpenAITools() []openai.Tool {
	toolList := e.registry.List()
	openAITools := make([]openai.Tool, 0, len(toolList))

	for _, name := range toolList {
		toolInst, err := e.registry.Get(name)
		if err != nil {
			continue
		}
		openAITools = append(openAITools, openai.Tool{
			Type: "function",
			Function: &openai.FunctionDefinition{
				Name:        toolInst.Name(),
				Description: toolInst.Description(),
				Parameters:  toolInst.Parameters(),
			},
		})
	}

	return openAITools
}

// formatObservations 将工具调用结果格式化为可读的观察文本。
func (e *AgentEngine) formatObservations(calls []types.ToolCall) string {
	var parts []string
	for _, tc := range calls {
		if tc.Result != nil && tc.Result.Success {
			parts = append(parts, fmt.Sprintf("[%s]: %s", tc.Name, truncate(tc.Result.Output, 200)))
		}
	}
	return strings.Join(parts, "\n")
}

// sendEvent 发送流式事件的快捷方法。
func (e *AgentEngine) sendEvent(ch chan<- types.StreamEvent, eventType types.StreamEventType, content string, index, total int, done bool) {
	ch <- types.StreamEvent{
		Type:      eventType,
		Content:   content,
		Index:     index,
		StepCount: total,
		Done:      done,
	}
}

// 工具函数
func truncate(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "..."
}

// splitIntoChunks 将字符串按 rune 拆分成指定大小的块，用于流式输出。
func splitIntoChunks(text string, chunkSize int) []string {
	if text == "" {
		return nil
	}
	runes := []rune(text)
	if chunkSize <= 0 {
		chunkSize = 3
	}
	chunks := make([]string, 0, (len(runes)+chunkSize-1)/chunkSize)
	for i := 0; i < len(runes); i += chunkSize {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}
	return chunks
}
