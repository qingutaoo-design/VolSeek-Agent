// Package types 定义 VolSeek-Agent 的全部核心数据类型。
// 所有其他包都依赖此包，因此它必须保持零外部依赖（仅标准库）。
package types

import (
	"encoding/json"
	"time"
)

// ============================================================================
// Agent 配置
// ============================================================================

// AgentConfig 是 Agent 引擎的完整配置。
// Temperature: LLM 的创造性程度 (0.0~2.0)，越低越确定
// MaxPlanningRounds: 最大规划轮次，防止无限循环
// EnableReflection: 是否启用自我反思（答案自检）
// EnableGraphRAG: 是否启用知识图谱增强检索
// MaxContextTokens: 上下文窗口预算，超过后触发压缩
type AgentConfig struct {
	Temperature       float64  `json:"temperature"`
	MaxPlanningRounds int      `json:"max_planning_rounds"`
	EnableReflection  bool     `json:"enable_reflection"`
	EnableGraphRAG    bool     `json:"enable_graph_rag"`
	WebSearchEnabled  bool     `json:"web_search_enabled"`
	ParallelToolCalls bool     `json:"parallel_tool_calls"`
	AllowedTools      []string `json:"allowed_tools"`
	MaxContextTokens  int      `json:"max_context_tokens"`
}

// ============================================================================
// Agent 状态和执行记录
// ============================================================================

// AgentState 跟踪一次 Agent 执行的完整状态。
// Plan: Agent 制定的执行计划
// Steps: 已经执行完成的步骤
// IsComplete: 是否完成
// FinalAnswer: 最终的答案
// Confidence: 答案的置信度 (0.0~1.0)
type AgentState struct {
	Plan        *Plan       `json:"plan,omitempty"`
	Steps       []Step      `json:"steps"`
	IsComplete  bool        `json:"is_complete"`
	FinalAnswer string     `json:"final_answer,omitempty"`
	Confidence  float64     `json:"confidence,omitempty"`
	Sources     []SourceRef `json:"sources,omitempty"`
}

// Plan 是 Agent 在开始执行前制定的结构化计划。
// 每个 Plan 包含多个顺序执行的 Step。
type Plan struct {
	Goal        string   `json:"goal"`         // 用户问题的重述
	SubGoals    []string `json:"sub_goals"`    // 子目标列表
	RequiresTool bool    `json:"requires_tool"` // 是否需要调用工具
	Reasoning   string   `json:"reasoning"`    // 规划时的推理过程
}

// Step 记录 Agent 执行的一个步骤（一次 Think-Act-Observe 循环）。
type Step struct {
	Index       int        `json:"index"`
	Goal        string     `json:"goal"`          // 本步骤的目标
	Thought     string     `json:"thought"`       // 本步骤的思考过程
	ToolCalls   []ToolCall `json:"tool_calls"`    // 本步骤调用的工具
	Observation string     `json:"observation"`   // 工具调用的观察结果
	Timestamp   time.Time  `json:"timestamp"`
}

// ToolCall 记录一次工具调用。
type ToolCall struct {
	ID       string         `json:"id"`
	Name     string         `json:"name"`
	Args     json.RawMessage `json:"args"`
	Result   *ToolResult    `json:"result"`
	Duration time.Duration  `json:"duration"`
}

// ToolResult 是工具执行的结果。
// Success: 是否成功
// Output: 文本输出（用于 LLM 消费）
// Data: 结构化数据（用于程序消费）
// Images: Base64 编码的图片数据
type ToolResult struct {
	Success bool                   `json:"success"`
	Output  string                 `json:"output"`
	Data    map[string]interface{} `json:"data,omitempty"`
	Error   string                 `json:"error,omitempty"`
	Images  []string               `json:"images,omitempty"`
}

// SourceRef 记录答案的来源引用。
type SourceRef struct {
	Title    string  `json:"title"`
	Content  string  `json:"content"`
	Source   string  `json:"source"` // "knowledge_base" | "web" | "graph"
	URL      string  `json:"url,omitempty"`
	Score    float64 `json:"score,omitempty"`
}

// ============================================================================
// LLM 相关类型
// ============================================================================

// LLMMessage 表示发送给 LLM 的消息。
// Role: "system" | "user" | "assistant" | "tool"
// Content: 消息内容
// ToolCalls: 当 Role=assistant 时，包含工具调用
// ToolCallID: 当 Role=tool 时，关联的调用 ID
type LLMMessage struct {
	Role       string          `json:"role"`
	Content    string          `json:"content"`
	ToolCalls  []ToolCallDef   `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

// ToolCallDef 是 LLM 返回的工具调用定义（JSON 格式）。
type ToolCallDef struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Name      string `json:"function_name"`
	Arguments string `json:"arguments"` // JSON 字符串
}

// LLMConfig LLM 客户端配置。
type LLMConfig struct {
	APIKey      string  `json:"api_key"`
	BaseURL     string  `json:"base_url"`
	Model       string  `json:"model"`
	Temperature float64 `json:"temperature"`
}

// ============================================================================
// RAG（检索增强生成）相关类型
// ============================================================================

// Chunk 是文档的一个分块。
type Chunk struct {
	ID        string            `json:"id"`
	Content   string            `json:"content"`
	Index     int               `json:"index"`
	DocTitle  string            `json:"doc_title"`
	Embedding []float64         `json:"-"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// ChunkConfig 分块配置。
type ChunkConfig struct {
	Size    int // 每块字符数
	Overlap int // 重叠字符数
}

// SearchResult 检索结果。
type SearchResult struct {
	Chunk   *Chunk  `json:"chunk"`
	Score   float64 `json:"score"`
	Method  string  `json:"method"` // "vector" | "keyword" | "hybrid" | "graph"
}

// KnowledgeGraph 知识图谱。
// Nodes: 实体节点
// Edges: 实体间的关系边
type KnowledgeGraph struct {
	Nodes []*EntityNode `json:"nodes"`
	Edges []*Relation   `json:"edges"`
}

// EntityNode 知识图谱中的实体节点。
type EntityNode struct {
	ID       string   `json:"id"`
	Name     string   `json:"name"`
	Type     string   `json:"type"` // "person" | "organization" | "concept" | "technology"
	Context  string   `json:"context"` // 该实体的上下文描述
	ChunkIDs []string `json:"chunk_ids,omitempty"` // 关联的文档分块
}

// Relation 实体间的关系。
type Relation struct {
	SourceID string `json:"source_id"`
	TargetID string `json:"target_id"`
	Relation string `json:"relation"` // "develops" | "uses" | "part-of" | ...
	Weight   float64 `json:"weight"`
}

// ============================================================================
// 查询路由相关类型
// ============================================================================

// QueryType 查询类型枚举。
type QueryType string

const (
	QueryFactual     QueryType = "factual"      // 事实性查询："什么是RAG？"
	QueryConceptual  QueryType = "conceptual"   // 概念性查询："解释一下注意力机制"
	QueryComparative QueryType = "comparative"  // 比较性查询："A和B有什么区别？"
	QueryProcedural  QueryType = "procedural"   // 步骤性查询："如何安装Go？"
	QueryAnalytical  QueryType = "analytical"   // 分析性查询："分析这个方案的优缺点"
	QueryRecent      QueryType = "recent"       // 实时性查询："今天比特币价格"
	QueryUnknown     QueryType = "unknown"      // 无法分类
)

// QueryIntent 描述用户查询的意图。
type QueryIntent struct {
	Type      QueryType `json:"type"`
	SubType   string    `json:"sub_type,omitempty"`
	Keywords  []string  `json:"keywords"`
	Entities  []string  `json:"entities"`
	NeedsWeb  bool      `json:"needs_web"`  // 是否需要联网搜索
	NeedsKG   bool      `json:"needs_kg"`   // 是否需要知识图谱
	Summary   string    `json:"summary"`    // 查询意图的文本总结
}

// ============================================================================
// 流式事件
// ============================================================================

// StreamEventType 流式事件类型。
type StreamEventType string

const (
	EventPlan       StreamEventType = "plan"         // 规划阶段
	EventThink      StreamEventType = "think"        // 思考过程
	EventToolCall   StreamEventType = "tool_call"    // 工具调用中
	EventToolResult StreamEventType = "tool_result"  // 工具调用结果
	EventAnswer     StreamEventType = "answer"       // 答案生成中
	EventReflect    StreamEventType = "reflect"      // 自我反思
	EventDone       StreamEventType = "done"         // 执行完成
	EventError      StreamEventType = "error"        // 错误
)

// StreamEvent 流式事件，用于 SSE 推送。
type StreamEvent struct {
	Type      StreamEventType `json:"type"`
	Content   string          `json:"content"`
	Index     int             `json:"index,omitempty"`
	StepCount int             `json:"step_count,omitempty"`
	Data      interface{}     `json:"data,omitempty"`
	Done      bool            `json:"done,omitempty"`
}
