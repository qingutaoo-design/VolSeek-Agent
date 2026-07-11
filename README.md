# 🚀 VolSeek-Agent 完全教学指南

> **从零构建一个具有规划、执行和反思能力的智能 RAG Agent**  
> **项目地址**: https://github.com/qingutaoo-design/VolSeek-Agent  
> **技术栈**: Go + OpenAI/DeepSeek API + 向量检索 + 知识图谱  
> **学习周期**: 5个部分

---

## 📋 目录

1. [项目概述与架构设计](#1-项目概述与架构设计)
2. [快速启动指南](#2-快速启动指南)
3. [Part 1 — 核心类型系统](#3-part-1--核心类型系统)
4. [Part 2 — LLM 客户端封装](#4-part-2--llm-客户端封装)
5. [Part 3 — RAG 引擎（向量搜索 + 知识图谱）](#5-part-3--rag-引擎向量搜索--知识图谱)
6. [Part 4 — Agent 引擎（Plan-Execute-Reflect）](#6-part-4--agent-引擎plan-execute-reflect)
7. [Part 5 — 工具系统与主入口](#7-part-5--工具系统与主入口)
8. [创新亮点总结](#8-创新亮点总结)


---

## 1. 项目概述与架构设计

### 1.1 这个项目做了什么？

**VolSeek-Agent** 是一个从零构建的智能 RAG Agent 框架。与市面上大多数 RAG 项目不同，它实现了 **三个核心创新**：

| 创新点 | 说明 | 对比传统 ReAct |
|--------|------|---------------|
| **Plan-then-Execute** | Agent 先制定结构化计划，再按计划执行 | 传统 ReAct 边想边做，容易跑偏 |
| **GraphRAG** | 构建知识图谱理解实体关系，支持关系遍历查询 | 普通 RAG 只做向量搜索，不懂"关系" |
| **Self-Reflection** | 生成答案后自我评审质量，自动修正 | 普通 RAG 一次生成不改 |

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VolSeek-Agent                               │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │   types   │  │   llm    │  │   rag    │  │      tools        │   │
│  │ (类型定义) │  │ (LLM调用) │  │ (RAG引擎) │  │   (工具注册中心)    │   │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────────┘   │
│                                        │                            │
│                                        ▼                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    agent/engine.go                            │   │
│  │  ┌───────┐    ┌──────────┐    ┌──────────┐                   │   │
│  │  │ Plan  │───▶│ Execute  │───▶│ Reflect  │                   │   │
│  │  │(规划) │    │  (执行)   │    │  (反思)   │                   │   │
│  │  └───────┘    └──────────┘    └──────────┘                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    cmd/main.go                                │   │
│  │         交互入口 + 示例文档索引 + 流式输出                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 项目结构

```
VolSeek-Agent/
├── cmd/main.go                 # 入口（交互模式 + 命令行模式）
├── internal/
│   ├── types/types.go          # 全部核心类型定义
│   ├── llm/client.go           # LLM 客户端（Chat/Embed/Stream）
│   ├── rag/rag.go              # RAG 引擎（Chunker/Retriever/Graph/QueryRouter）
│   ├── tools/tools.go          # 工具系统（注册中心 + 5 个内置工具）
│   ├── agent/engine.go         # Agent 引擎（Plan-Execute-Reflect）
│   └── config/config.go        # 环境变量配置
├── .env.example                # 环境变量模板
└── go.mod
```

---

## 2. 快速启动指南

> 从零开始运行 VolSeek-Agent，最快只需 2 分钟。

### 2.1 前置条件

| 运行方式 | 要求 |
|---------|------|
| **源代码运行** | Go 1.21+、Git |
| **预编译二进制** | Windows（`main.exe` 已附在项目根目录） |
| **API 密钥** | 至少一个兼容 OpenAI API 的服务商（见下方对照表） |

### 2.2 选择你的服务商方案

VolSeek-Agent 需要 **聊天 API** 和 **Embedding（向量化）API**。不同服务商的策略不同：

| 方案 | 聊天 API | Embedding API | 费用 | 推荐指数 |
|------|---------|--------------|------|---------|
| **🥇 OpenAI 一站式** | `gpt-4o-mini` | `text-embedding-ada-002` | 付费，但最稳定 | ⭐⭐⭐⭐⭐ |
| **🥈 DeepSeek + SiliconFlow（免费）** | `deepseek-v4-flash` | `BAAI/bge-large-zh-v1.5` | **聊天付费，Embedding 免费** | ⭐⭐⭐⭐ |
| **🥉 Ollama 全本地** | `llama3` 等 | `nomic-embed-text` | **完全免费**（需本地 GPU） | ⭐⭐⭐ |

> 💡 **本文推荐方案 🥈**：聊天用你已有的 DeepSeek 密钥 + 向量化用 SiliconFlow 的免费赠金。下面以此为例。

---

### 2.3 快速上手（以 DeepSeek + text-embedding-v4 为例）

#### 第一步：获取项目

```bash
git clone https://github.com/qingutaoo-design/VolSeek-Agent.git
cd VolSeek-Agent
```

#### 第二步：注册 SiliconFlow（获取免费 Embedding API）

DeepSeek **不支持**向量化，需要另一个服务商来做。百炼大模型 注册即送 **100w token**（无需绑卡）：

1. 打开 https://bailian.console.aliyun.com 注册账号
2. 进入 **API 密钥** 页面，创建一个密钥（以 `sk-` 开头）
3. 复制密钥，下一步使用

#### 第三步：配置环境变量

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

编辑 `.env` 文件，按你的服务商填写。关键是要配 **两组** 配置——聊天和向量化可能是不同的服务商：

**场景 A：DeepSeek 聊天 + text-embedding-v4 向量化（推荐免费方案）**

```ini
# ─── 聊天 API（DeepSeek）───────────────────────────────────
LLM_API_KEY=sk-your-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-flash

# ─── Embedding 向量化 API（text-embedding-v4 免费）─────────────
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
```

**场景 B：OpenAI 一站式（一个密钥全搞定）**

```ini
# ─── 聊天 API（OpenAI）──────────────────────────────────────
LLM_API_KEY=sk-your-openai-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# ─── Embedding（保持默认即可）──────────────────────────────
# EMBEDDING_BASE_URL=              ← 留空，复用 LLM_BASE_URL
# EMBEDDING_MODEL=text-embedding-ada-002
```

**场景 C：Ollama 全本地运行（完全免费，需本地部署）**

```ini
# ─── 聊天 API（Ollama 本地）────────────────────────────────
LLM_API_KEY=ollama                    # Ollama 不需要真实密钥
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3

# ─── Embedding 向量化（Ollama 本地）───────────────────────
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
```

> **注意**：场景 C 需要先运行 `ollama pull llama3` 和 `ollama pull nomic-embed-text`。

#### 第四步：运行

**方式 A：使用预编译二进制（Windows，无需安装 Go）**

```bash
# 交互模式
.\main.exe

# 直接提问
.\main.exe "RAG 和 GraphRAG 有什么区别？"
```

**方式 B：源码运行**

```bash
# 交互模式
go run cmd/main.go

# 直接提问
go run cmd/main.go "什么是 RAG？"
```

**方式 C：编译后运行**

```bash
go build -o volseek cmd/main.go

# Linux / macOS
./volseek "Go 语言的 goroutine 是什么？"

# Windows
.\volseek.exe "Go 语言的 goroutine 是什么？"
```

---

### 2.4 首次运行体验

启动后，你将看到如下输出：

```
🔄 Initializing LLM client... ✅
🔄 Initializing RAG engine... ✅
🔄 Indexing sample documents... ✅ (10 chunks, 15 entities, 12 relations)
🔄 Initializing tools... ✅
🔄 Initializing Agent engine... ✅
============================================================
🤖 VolSeek-Agent 已就绪！
============================================================

💡 输入你的问题（输入 'exit' 退出, 'stats' 查看状态）:
```

项目已内置 3 篇示例文档（RAG 技术介绍、Go 语言入门、VolSeek-Agent 设计文档），启动后自动索引，可直接提问。

#### 示例问答

```
❓ RAG 和 GraphRAG 有什么区别？

📋 Analyzing your question...

🤔 我来分析 RAG 和 GraphRAG 的区别...

🔧 🔍 Searching knowledge base...

📝 RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的...
GraphRAG 是 RAG 的进阶版本，通过构建知识图谱来理解实体间的关系...

🎯 Confidence: 92%
```

其他可以尝试的问题：

| 问题 | 预期效果 |
|------|---------|
| `什么是 RAG？` | 事实性查询，精确检索文档块 |
| `Go 语言的 goroutine 是什么？` | 概念性查询，语义检索 |
| `RAG 和 GraphRAG 有什么区别？` | 比较性查询，混合检索 + RRF 融合 |
| `分析 RAG 技术的优缺点` | 分析性查询，向量搜索 + 知识图谱拓展 |
| `stats` | 查看系统状态（文档块数、实体数、关系数） |

---

### 2.5 配置详解

`.env` 中所有可配置项：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | — | **必填**。聊天 API 的密钥 |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | 聊天 API 的端点地址 |
| `LLM_MODEL` | `gpt-4o-mini` | 聊天模型名称 |
| `EMBEDDING_BASE_URL` | `（同 LLM_BASE_URL）` | Embedding API 端点。**不同服务商时需单独设置** |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | Embedding 模型名称 |
| `SERP_API_KEY` | — | 网页搜索 API 密钥（可选，留空不影响核心功能） |

#### 常见 Embedding 模型推荐

| 服务商 | 推荐模型 | 特点 |
|-------|---------|------|
| OpenAI | `text-embedding-ada-002` | 通用，维数 1536 |
| OpenAI | `text-embedding-3-small` | 性价比更高 |
| **SiliconFlow（免费）** | **`BAAI/bge-large-zh-v1.5`** | **中文优化，推荐** |
| SiliconFlow（免费） | `Qwen/Qwen3-Embedding-0.6B` | 最便宜，$0.01/M tokens |
| Ollama 本地 | `nomic-embed-text` | 完全免费，需本地运行 |

---

### 2.6 常见问题排查

#### ❌ 启动时报 "embedding failed" / 启动卡住

**原因**：你的 `LLM_BASE_URL` 指向的 API 不支持 Embedding（如 DeepSeek）。

**解决**：在 `.env` 中单独配置 `EMBEDDING_BASE_URL`，指向支持 Embedding 的服务商（如 SiliconFlow）。

```ini
# 加入这两行即可
EMBEDDING_BASE_URL=https://api.siliconflow.com/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

#### ❌ "401 Unauthorized" / "鉴权失败"

**原因**：API 密钥不正确，或用了 OpenAI 的密钥去调 DeepSeek。

**解决**：检查 `.env` 中 `LLM_API_KEY` 和 `LLM_BASE_URL` 是否匹配同一服务商。

#### ❌ "模型不存在" 或 "404"

**原因**：`LLM_MODEL` 或 `EMBEDDING_MODEL` 名称在当前服务商中不存在。

**解决**：
- 查一下对应服务商的官方文档确认模型名
- DeepSeek 常用：`deepseek-chat` / `deepseek-v3` / `deepseek-v4-flash`
- SiliconFlow 模型列表：https://docs.siliconflow.com/en/api-reference/embeddings/create-embeddings

#### ❌ 程序能运行但回答质量差

**原因**：内置文档是中文示例，如果用了英文 Embedding 模型可能匹配不准。

**解决**：推荐使用中文优化的模型，如 `BAAI/bge-large-zh-v1.5`。

---

### 2.7 常用命令速查

```bash
# 交互模式
go run cmd/main.go

# 单次查询
go run cmd/main.go "你的问题"

# 编译为二进制
go build -o volseek cmd/main.go

# 生产部署（关闭反思可提升速度）
go build -o volseek -ldflags="-s -w" cmd/main.go

# 查看系统统计（交互模式下输入）
stats
```

---

### 2.8 下一步

- 阅读 **[Part 1 — 核心类型系统](#3-part-1--核心类型系统)**，了解项目的数据类型设计
- 将你自己的文档加入 `cmd/main.go` 的 `indexSampleDocuments` 函数，构建私有知识库
- 调整 `AgentConfig` 中的参数（如 `EnableReflection`、`ParallelToolCalls`），观察不同配置下的表现差异

---

## 2. Part 1 — 核心类型系统

### 📂 文件: `internal/types/types.go`

核心类型是整个项目的**数据骨架**。所有包都依赖于此，因此它必须保持零外部依赖。

#### 2.1 Agent 配置

```go
// AgentConfig 整个 Agent 的运行配置。
// Temperature: LLM 创造性（0.0~2.0），越低回答越确定
// MaxPlanningRounds: 最大规划轮次，防止无限循环
// EnableReflection: 是否在回答后做自我质量评审
// EnableGraphRAG: 是否启用知识图谱增强检索
type AgentConfig struct {
    Temperature       float64  // LLM 温度参数
    MaxPlanningRounds int     // 最大循环次数
    EnableReflection  bool    // 启用自我反思
    EnableGraphRAG    bool    // 启用知识图谱
    WebSearchEnabled  bool    // 启用网页搜索
    ParallelToolCalls bool    // 启用并行工具调用
    AllowedTools      []string // 允许的工具列表
    MaxContextTokens  int     // 上下文窗口上限
}
```

**设计思考**：我们把配置集中在一个结构体中，避免散落在各个函数参数里。这样传递配置变得非常清晰。

#### 2.2 Agent 状态与执行记录

```go
// AgentState 追踪一次 Agent 执行的完整状态。
// Steps 记录每一步的详细信息，用于调试和展示。
type AgentState struct {
    Plan        *Plan       // 执行计划
    Steps       []Step      // 执行步骤历史
    IsComplete  bool        // 是否完成
    FinalAnswer string      // 最终答案
    Confidence  float64     // 置信度 (0.0~1.0)
    Sources     []SourceRef // 来源引用列表
}

// Plan Agent 在开始执行前制定的结构化计划。
// 这就是"Plan-then-Execute"的 Plan 部分。
type Plan struct {
    Goal        string   // 目标重述（把用户问题转化为清晰目标）
    SubGoals    []string // 子目标列表（步骤分解）
    RequiresTool bool    // 是否需要调用工具
    Reasoning   string   // 规划时的推理过程
}

// Step 记录一次 Think-Act-Observe 循环。
type Step struct {
    Index       int        // 步索引
    Goal        string     // 本步目标
    Thought     string     // LLM 的思考内容
    ToolCalls   []ToolCall // 调用的工具
    Observation string     // 观察结果
    Timestamp   time.Time  // 时间戳
}
```

#### 2.3 LLM 消息类型

```go
// LLMMessage 是发送给 LLM 的消息，支持 OpenAI 格式。
// Role 可以是 "system" / "user" / "assistant" / "tool"
// ToolCalls 字段仅在 Role=assistant 时使用
// ToolCallID 字段仅在 Role=tool 时使用
type LLMMessage struct {
    Role       string      // 角色
    Content    string      // 内容
    ToolCalls  []ToolCallDef // 工具调用（assistant 消息）
    ToolCallID string      // 工具调用 ID（tool 消息）
    Name       string      // 工具名称（tool 消息）
}
```

#### 2.4 RAG 类型

```go
// Chunk 文档分块的核心结构。
// Embedding 字段不会序列化到 JSON（`json:"-"`），避免存储臃肿。
type Chunk struct {
    ID        string            // 唯一标识
    Content   string            // 分块内容
    Index     int               // 在文档中的序号
    DocTitle  string            // 所属文档标题
    Embedding []float64         `json:"-"` // 向量（不序列化）
    Metadata  map[string]string // 元数据
}

// KnowledgeGraph 知识图谱：实体 + 关系。
type KnowledgeGraph struct {
    Nodes []*EntityNode  // 实体节点
    Edges []*Relation    // 关系边
}

// EntityNode 图谱中的实体。
type EntityNode struct {
    ID       string   // 唯一标识
    Name     string   // 实体名称，如 "RAG", "Go 语言"
    Type     string   // 类型: "person" | "concept" | "technology"
    Context  string   // 上下文描述
    ChunkIDs []string // 关联的文档分块 ID
}
```

#### 2.5 查询意图路由类型

```go
// QueryType 定义了 7 种查询类型，每种对应不同的检索策略。
type QueryType string
const (
    QueryFactual     QueryType = "factual"     // 事实性："什么是RAG？"
    QueryConceptual  QueryType = "conceptual"  // 概念性："解释注意力机制"
    QueryComparative QueryType = "comparative" // 比较性："A和B的区别"
    QueryProcedural  QueryType = "procedural"  // 步骤性："如何安装Go？"
    QueryAnalytical  QueryType = "analytical"  // 分析性："分析优缺点"
    QueryRecent      QueryType = "recent"      // 实时性："今天行情"
    QueryUnknown     QueryType = "unknown"     // 无法分类
)

// QueryIntent 描述查询的完整意图。
type QueryIntent struct {
    Type     QueryType  // 查询类型
    Keywords []string   // 关键词
    Entities []string   // 涉及实体
    NeedsWeb bool       // 需要联网
    NeedsKG  bool       // 需要知识图谱
    Summary  string     // 意图总结
}
```

**创新点**：这就是"自适应路由"的基础——我们先理解用户想要什么，再选择合适的检索策略。

#### 2.6 流式事件

```go
// StreamEvent 所有流式事件的统一格式。
// 客户端读取这些事件来展示 Agent 的思考过程。
type StreamEvent struct {
    Type      StreamEventType  // 事件类型
    Content   string           // 内容
    Index     int              // 当前步号
    StepCount int              // 总步数
    Data      interface{}      // 附加数据
    Done      bool             // 是否结束
}
```

---

## 3. Part 2 — LLM 客户端封装

### 📂 文件: `internal/llm/client.go`

#### 3.1 客户端初始化

```go
// NewClient 创建 LLM 客户端。
// 通过环境变量配置 API 端点，支持 OpenAI / DeepSeek / Azure 等兼容 API。
func NewClient(apiKey, baseURL, model string) *Client {
    cfg := openai.DefaultConfig(apiKey)
    cfg.BaseURL = baseURL
    return &Client{
        client: openai.NewClientWithConfig(cfg),
        model:  model,
    }
}
```

支持任意兼容 OpenAI API 格式的服务商，只需修改 `LLM_BASE_URL`：
- OpenAI: `https://api.openai.com/v1`
- DeepSeek: `https://api.deepseek.com/v1`
- 本地 Ollama: `http://localhost:11434/v1`

#### 3.2 带重试的聊天补全

```go
// ChatWithRetry 执行带自动重试的聊天补全。
// 对 429/5xx/超时 等临时性错误最多重试 3 次。
func (c *Client) ChatWithRetry(ctx context.Context, 
    messages []types.LLMMessage, 
    tools []openai.Tool, 
    temperature float64,
) (*ChatResponse, error) {
    maxRetries := 3
    var lastErr error

    for attempt := 0; attempt < maxRetries; attempt++ {
        if attempt > 0 {
            time.Sleep(time.Duration(attempt+1) * time.Second) // 递增等待
        }
        result, err := c.Chat(ctx, messages, tools, temperature)
        if err == nil {
            return result, nil
        }
        lastErr = err
        if !isRetryable(err) {
            return nil, err // 非临时错误不重试
        }
    }
    return nil, fmt.Errorf("LLM call failed after retries: %w", lastErr)
}
```

#### 3.3 Embedding（向量化）

```go
// EmbedBatch 批量向量化文本。
// 注意 OpenAI 返回的向量是 float32，需要转为 float64 供计算使用。
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
    const batchSize = 20
    results := make([][]float64, len(texts))

    for i := 0; i < len(texts); i += batchSize {
        end := i + batchSize
        if end > len(texts) { end = len(texts) }
        batch := texts[i:end]

        resp, err := c.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
            Model: openai.EmbeddingModel("text-embedding-ada-002"),
            Input: batch,
        })
        if err != nil {
            return nil, fmt.Errorf("batch embedding failed: %w", err)
        }
        for _, data := range resp.Data {
            if data.Index < len(batch) {
                results[i+data.Index] = float32to64(data.Embedding) // 关键转换
            }
        }
        time.Sleep(200 * time.Millisecond) // API 限速保护
    }
    return results, nil
}
```

**注意**：OpenAI 的 Embedding API 返回 `[]float32`，但我们的余弦相似度计算用 `[]float64`，必须做类型转换。

---

## 4. Part 3 — RAG 引擎（向量搜索 + 知识图谱）

### 📂 文件: `internal/rag/rag.go`

这是项目最厚的文件（~1000行），包含四个主要组件。

#### 4.1 Chunker — 智能文档分块器

```go
// Chunk 自动选择分块策略。
// 检测到 Markdown（# 标题）时用段落分块，保留语义完整性；
// 纯文本用固定大小重叠分块。
func (c *Chunker) Chunk(text, title string) []*types.Chunk {
    if strings.Contains(text, "# ") {
        return c.chunkByMarkdown(text, title)  // Markdown → 按段落
    }
    return c.chunkByFixedSize(text, title)      // 纯文本 → 固定大小
}
```

##### 固定大小分块（带智能截断）

```go
func (c *Chunker) chunkByFixedSize(text, title string) []*types.Chunk {
    for start < totalLen {
        end := start + c.config.Size
        
        // 智能截断：在句子边界处截断，而非生硬切断
        if end < totalLen {
            for j := len(chunkRunes) - 1; j > start+c.config.Size/2; j-- {
                if chunkRunes[j] == '.' || chunkRunes[j] == '。' {
                    end = start + j + 1  // 在句号处截断
                    break
                }
            }
        }
        
        // 滑动窗口（带 50 字符重叠）
        nextStart := end - c.config.Overlap
    }
}
```

##### Markdown 分块（按标题分段）

```go
func (c *Chunker) chunkByMarkdown(text, title string) []*types.Chunk {
    for _, line := range lines {
        // 遇到 Markdown 标题 (# ## ###) 时，刷新当前段落
        if strings.HasPrefix(trimmed, "# ") {
            flush()  // 保存当前段落，开始新段落
            currentHeading = strings.TrimLeft(trimmed, "# ")
        }
        currentBuilder.WriteString(line + "\n")
    }
}
```

**为什么需要两种分块策略？** Markdown 按标题分段保留语义完整性（一个段落不会被切开），纯文本按固定大小保证检索粒度（段落太长时降低命中率）。

#### 4.2 VectorStore — 内存向量存储

```go
type VectorStore struct {
    mu     sync.RWMutex       // 读写锁，支持并发
    chunks []*types.Chunk     // 所有分块
    docIndex map[string][]*types.Chunk // 按文档标题索引
}

// Search 执行暴力向量搜索（O(n)）。
// 计算所有向量与查询向量的余弦相似度，返回 Top-K。
func (vs *VectorStore) Search(queryEmbed []float64, topK int, threshold float64) []*types.SearchResult {
    for _, chunk := range vs.chunks {
        score := cosineSimilarity(queryEmbed, chunk.Embedding)
        if score >= threshold {
            results = append(results, &types.SearchResult{
                Chunk: chunk, Score: score, Method: "vector",
            })
        }
    }
    // 按分数降序，取 Top-K
    sort.Slice(results, func(i, j int) bool {
        return results[i].Score > results[j].Score
    })
    return results[:min(len(results), topK)]
}
```

**为什么用暴力搜索？** 对小规模数据（<10万条），暴力搜索的简单性和正确性优于近似搜索（HNSW/IVF）。生产环境应替换为 pgvector / Milvus。

#### 4.3 Retriever — 多策略检索器 ⭐ 核心创新

```go
// Retrieve 根据查询意图选择最优检索策略。
// 这就是"自适应路由"的核心实现。
func (r *Retriever) Retrieve(ctx context.Context, query string, intent *types.QueryIntent) ([]*types.SearchResult, error) {
    switch intent.Type {
    case types.QueryFactual:
        // 事实性查询 → 关键词 + 向量混合搜索
        return r.hybridSearch(ctx, query, r.topK)
        
    case types.QueryConceptual:
        // 概念性查询 → 纯向量搜索（语义相似度）
        embed, _ := r.llm.Embed(ctx, query)
        return r.store.Search(embed, r.topK, 0.5)
        
    case types.QueryComparative:
        // 比较性查询 → 混合搜索 + RRF 融合
        results := r.hybridSearch(ctx, query, r.topK*2)
        return applyRRF(results, 60)  // RRF 常数 60
        
    case types.QueryAnalytical:
        // 分析性查询 → 向量搜索 + 知识图谱拓展
        vecResults := r.store.Search(embed, r.topK, 0.5)
        graphResults := r.expandWithGraph(vecResults)  // 🔥 图谱拓展
        return append(vecResults, graphResults...)
        
    case types.QueryRecent:
        // 实时性查询 → 不检索（等 web_search 工具）
        return nil, nil
    }
}
```

**什么是 RRF（Reciprocal Rank Fusion）？**
```go
// applyRRF 融合多个排序结果。
// 核心思想：在每个排序列表中排得越靠前，综合得分越高。
// RRF = Σ(1/(k + rank))，k 通常取 60。
func applyRRF(results []*types.SearchResult, k int) []*types.SearchResult {
    rankMap := make(map[string]map[string]int) // method -> chunkID -> rank
    for i, res := range results {
        if _, ok := rankMap[res.Method]; !ok {
            rankMap[res.Method] = make(map[string]int)
        }
        rankMap[res.Method][res.Chunk.ID] = i + 1
    }
    // 计算 RRF 分数
    for id, res := range unique {
        var rrf float64
        for method, ranks := range rankMap {
            if rank, ok := ranks[id]; ok {
                rrf += 1.0 / float64(k + rank)  // 关键公式
            }
        }
        entries = append(entries, entry{result: res, rrf: rrf})
    }
}
```

#### 4.4 GraphStore — 知识图谱存储 🔥 核心创新

```go
// GetNeighbors 获取指定实体的 N 跳邻居。
// 使用 DFS 遍历关系图，用于"沿着关系找信息"。
func (gs *GraphStore) GetNeighbors(entityID string, hops int) []*types.EntityNode {
    visited := make(map[string]bool)
    var result []*types.EntityNode

    var dfs func(id string, depth int)
    dfs = func(id string, depth int) {
        if depth > hops || visited[id] { return }
        visited[id] = true
        
        for _, rel := range gs.relations {
            var neighborID string
            if rel.SourceID == id {
                neighborID = rel.TargetID
            } else if rel.TargetID == id {
                neighborID = rel.SourceID
            } else { continue }
            
            if neighbor := gs.entities[neighborID]; neighbor != nil {
                result = append(result, neighbor)
                dfs(neighborID, depth+1)
            }
        }
    }
    dfs(entityID, 0)
    return result
}
```

**GraphRAG 的应用场景**：当用户问"RAG 和 GraphRAG 有什么区别？"时，向量搜索找到相关文档，知识图谱沿着"RAG"→"进阶版本"→"GraphRAG"的关系路径找到更多上下文。

#### 4.5 QueryRouter — 查询意图路由器 🔥 核心创新

```go
// Analyze 用 LLM 分析查询意图。
// 低温度（0.1）确保分类结果稳定。
func (qr *QueryRouter) Analyze(ctx context.Context, query string) (*types.QueryIntent, error) {
    systemPrompt := `Classify the question into one of:
    - factual: 事实/定义
    - conceptual: 概念解释
    - comparative: 比较
    - procedural: 步骤
    - analytical: 分析
    - recent: 实时信息

    Respond with JSON: {"type": "...", "needs_web": false, "needs_kg": false}`

    resp, err := qr.llm.Chat(ctx, messages, nil, 0.1)
    // 如果 LLM 分类失败，用启发式规则兜底
    if err != nil {
        return qr.heuristicAnalyze(query), nil
    }
    return qr.parseIntent(resp.Content), nil
}
```

**兜底方案**：当 LLM 不可用时，用关键词匹配做启发式分类（检测"什么是"→factual、"区别"→comparative 等）。

---

## 5. Part 4 — Agent 引擎（Plan-Execute-Reflect）

### 📂 文件: `internal/agent/engine.go`

这是项目的**心脏**，实现了 Plan-then-Execute 模式。

#### 5.1 三阶段架构

```go
// executeWithEvents 实际执行逻辑。
// 三个清晰的阶段：Plan → Execute → Reflect
func (e *AgentEngine) executeWithEvents(ctx context.Context, query string, ch chan<- types.StreamEvent) {
    state := &types.AgentState{}

    // ════════════════════════════════════════════════════════════
    // 阶段 1: 规划 (Plan)
    // ════════════════════════════════════════════════════════════
    e.sendEvent(ch, types.EventPlan, "Analyzing your question...", 0, 0, false)
    plan, _ := e.createPlan(ctx, query)
    state.Plan = plan

    // ════════════════════════════════════════════════════════════
    // 阶段 2: 执行 (Execute) — 核心循环
    // ════════════════════════════════════════════════════════════
    for stepIndex < maxRounds {
        // 1. Think: 调用 LLM
        resp, _ = e.llm.ChatWithRetry(ctx, messages, openAITools, e.config.Temperature)
        
        // 2. Analyze: 分析响应
        if len(resp.ToolCalls) > 0 {
            // 有工具调用 → 执行工具
            if tc.Name == "final_answer" {
                // LLM 主动提交最终答案
                break
            }
            step.ToolCalls = e.executeToolCalls(ctx, resp.ToolCalls, ch, stepIndex)
            // 追加结果到消息列表
            messages = append(messages, toolResultMessages...)
        } else if resp.FinishReason == "stop" && resp.Content != "" {
            // LLM 自然停止 → 直接给出答案
            state.FinalAnswer = resp.Content
            break
        }
    }

    // ════════════════════════════════════════════════════════════
    // 阶段 3: 反思 (Reflect) — 可选 🔥 核心创新
    // ════════════════════════════════════════════════════════════
    if e.config.EnableReflection && state.FinalAnswer != "" {
        refinedAnswer, confidence, _ := e.reflectOnAnswer(ctx, query, state.FinalAnswer)
        state.FinalAnswer = refinedAnswer  // 用改进后的答案替换
        state.Confidence = confidence
    }
}
```

#### 5.2 规划阶段详解

```go
// createPlan 让 LLM 生成结构化执行计划。
// 这是 Plan-then-Execute 的第一步：先想清楚怎么做。
func (e *AgentEngine) createPlan(ctx context.Context, query string) (*types.Plan, error) {
    // 1. 先用意图路由器分析查询
    intent, _ := e.router.Analyze(ctx, query)
    
    // 2. 让 LLM 生成 JSON 格式的计划
    planPrompt := `Analyze the question and create a plan.
    Respond with JSON: {"goal": "...", "sub_goals": [...], "requires_tool": true/false}`
    
    resp, err := e.llm.Chat(ctx, messages, nil, 0.3)
    
    // 3. 解析 JSON 计划
    plan := parsePlanJSON(resp.Content)
    if plan == nil {
        // JSON 解析失败 → 返回默认计划
        return &Plan{Goal: query, SubGoals: defaultSteps}
    }
    return plan, nil
}
```

**为什么先规划再执行？** 传统 ReAct 的"边想边做"模式容易让 LLM 进入循环（反复调用无用工具）。先规划明确目标，执行效率大幅提升。

#### 5.3 反思阶段详解 🔥

```go
// reflectOnAnswer 让 Agent 对自己的答案做质量评审。
// 评审维度包括准确性、完整性、清晰度。
func (e *AgentEngine) reflectOnAnswer(ctx context.Context, query, answer string) (string, float64, error) {
    reflectPrompt := `Review this Q&A pair:
    QUESTION: %s
    ANSWER: %s
    
    Evaluate on:
    1. Accuracy (0-1): 事实错误？
    2. Completeness (0-1): 完整回答问题？
    3. Clarity (0-1): 清晰易懂？
    
    Respond JSON: {"refined_answer": "...", "confidence": 0.0}`

    resp, _ := e.llm.Chat(ctx, messages, nil, 0.2)
    
    var result struct {
        RefinedAnswer string  `json:"refined_answer"`
        Confidence    float64 `json:"confidence"`
    }
    json.Unmarshal([]byte(jsonStr), &result)
    
    if result.RefinedAnswer != "" {
        return result.RefinedAnswer, result.Confidence, nil
    }
    return answer, 0.7, nil  // 解析失败则保持原答案
}
```

#### 5.4 并行工具执行

```go
// executeParallel 使用 sync.WaitGroup 实现并发工具调用。
// 注意：不使用 errgroup.WithContext，因为第一个工具失败不应取消其他工具。
func (e *AgentEngine) executeParallel(
    ctx context.Context, calls []types.ToolCallDef,
    results []types.ToolCall, ch chan<- types.StreamEvent, stepIndex int,
) {
    var mu sync.Mutex
    var wg sync.WaitGroup
    
    for i, tc := range calls {
        i, tc := i, tc
        wg.Add(1)
        go func() {
            defer wg.Done()
            result := e.executeSingle(ctx, tc, ch, stepIndex)
            mu.Lock()
            results[i] = result
            mu.Unlock()
        }()
    }
    wg.Wait()
}
```

**安全性设计**：我们特意用 `sync.WaitGroup` 而非 `errgroup.WithContext`。因为 `errgroup` 会在任一 goroutine 返回错误时自动取消 context，导致其他正在执行的工具被中断。我们的设计是"各自为战，互不影响"。

---

## 6. Part 5 — 工具系统与主入口

### 📂 文件: `internal/tools/tools.go`

#### 6.1 工具接口

```go
// Tool 是所有工具的通用接口。
type Tool interface {
    Name() string                // 工具名称（LLM 通过名称引用）
    Description() string         // 描述（LLM 理解工具用途）
    Parameters() json.RawMessage // JSON Schema 参数定义
    Execute(ctx context.Context, args json.RawMessage) *types.ToolResult
}
```

#### 6.2 FinalAnswerTool — 最终答案控制 🔥

```go
// FinalAnswerTool 是 Agent 提交最终答案的专用工具。
// 创新点：LLM 必须显式调用此工具来提交答案，确保不会"忘记"提供答案。
type FinalAnswerTool struct {
    name       string
    parameters json.RawMessage
}

func NewFinalAnswerTool() *FinalAnswerTool {
    return &FinalAnswerTool{
        name: "final_answer",
        parameters: json.RawMessage(`{
            "type": "object",
            "properties": {
                "answer":     {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "sources":    {"type": "array", "items": {"type": "string"}}
            },
            "required": ["answer", "confidence"]
        }`),
    }
}
```

**设计思考**：普通的 RAG 系统靠 LLM "自然停止"来表示答案结束。但我们发现 LLM 有时会"忘记"给出答案，或者给出答案后又画蛇添足。通过强制调用 `final_answer` 工具，我们精确控制了"何时给出答案"这个关键节点。

#### 6.3 CalculatorTool — 递归下降解析器

```go
// evalExpression 使用递归下降解析器求值算术表达式。
// 支持: +, -, *, /, (), 整数和小数。
// 这是生产可用的实现，无外部依赖。
func evalExpression(expr string) (float64, error) {
    p := &exprParser{input: expr}
    return p.parse()
}

// exprParser 实现递归下降解析。
// 文法: expr → term (('+' | '-') term)*
//       term → factor (('*' | '/') factor)*
//       factor → NUMBER | '(' expr ')'
type exprParser struct {
    input string
    pos   int
}

func (p *exprParser) parseExpr() (float64, error) {
    left, _ := p.parseTerm()
    for {
        op := p.peek()
        if op != '+' && op != '-' { break }
        p.advance()
        right, _ := p.parseTerm()
        if op == '+' { left += right } else { left -= right }
    }
    return left, nil
}
```

**为什么不用 `go/parser`？** 标准库的表达式解析器太重（需要完整的 Go 语法支持）。递归下降解析器轻量、无外部依赖、正确性可证明。

### 📂 文件: `cmd/main.go`

#### 6.4 完整的初始化流程

```go
func main() {
    // 1. 加载环境变量
    config.Load()
    apiKey := config.GetEnv("LLM_API_KEY", "")
    
    // 2. 初始化 LLM 客户端
    llmClient := llm.NewClient(apiKey, baseURL, model)
    
    // 3. 初始化 RAG 引擎
    chunker := rag.NewChunker(&types.ChunkConfig{Size: 500, Overlap: 50})
    vectorStore := rag.NewVectorStore()
    graphStore := rag.NewGraphStore()
    queryRouter := rag.NewQueryRouter(llmClient)
    retriever := rag.NewRetriever(vectorStore, llmClient, graphStore, 5)
    
    // 4. 索引示例文档（含 Embedding + 知识图谱构建）
    indexSampleDocuments(chunker, llmClient, vectorStore, graphStore)
    
    // 5. 初始化工具注册中心
    registry := tools.NewRegistry()
    registry.Register(tools.NewKnowledgeSearchTool(searchFn))
    registry.Register(tools.NewGraphSearchTool(graphSearchFn))
    registry.Register(tools.NewCalculatorTool())
    registry.Register(tools.NewFinalAnswerTool())
    
    // 6. 初始化 Agent 引擎
    volseek := agent.NewAgentEngine(&types.AgentConfig{
        Temperature:       0.7,
        MaxPlanningRounds: 10,
        EnableReflection:  true,   // 🔥 启用自我反思
        EnableGraphRAG:    true,   // 🔥 启用知识图谱
        ParallelToolCalls: true,   // 🔥 启用并行执行
    }, llmClient, registry, queryRouter, retriever)
    
    // 7. 进入交互模式
    runQuery(context.Background(), volseek, query)
}
```

#### 6.5 流式事件处理

```go
func runQuery(ctx context.Context, volseek *agent.AgentEngine, query string) {
    eventCh, _ := volseek.Execute(ctx, query)
    
    for event := range eventCh {
        switch event.Type {
        case types.EventPlan:
            fmt.Printf("📋 %s\n", event.Content)
        case types.EventThink:
            fmt.Printf("🤔 %s\n", event.Content)
        case types.EventToolCall:
            fmt.Printf("🔧 %s\n", event.Content)
        case types.EventToolResult:
            fmt.Printf("   %s\n", event.Content)
        case types.EventReflect:
            fmt.Printf("🔍 %s\n", event.Content)
        case types.EventAnswer:
            fmt.Printf("📝 %s\n", event.Content)
        case types.EventDone:
            fmt.Printf("🎯 Confidence: %.0f%%\n", confidence*100)
        }
    }
}
```

---

## 7. 创新亮点总结

### 7.1 Plan-then-Execute vs 传统 ReAct

| 维度 | 传统 ReAct | VolSeek Plan-then-Execute |
|------|-----------|--------------------------|
| 执行方式 | 边想边做 | 先计划后执行 |
| 可控性 | 低（易发散） | 高（目标明确） |
| 效率 | 低（可能反复） | 高（一次性规划） |
| 可解释性 | 中 | 高（计划清晰可见） |

### 7.2 GraphRAG 增强检索

```
用户问："RAG和GraphRAG的区别？"
         │
         ▼
  ┌──────────────┐
  │ 向量搜索找文档  │──→ 找到[RAG介绍]和[GraphRAG]两个文档块
  └──────────────┘
         │
         ▼
  ┌──────────────┐
  │ 知识图谱拓展   │──→ RAG → co-occurs-with → GraphRAG
  └──────────────┘     (找到更多关系上下文)
         │
         ▼
  ┌──────────────┐
  │ 综合两个来源   │──→ 更完整、更准确的答案
  └──────────────┘
```

### 7.3 自适应查询路由

| 查询类型 | 识别方式 | 检索策略 |
|---------|---------|----------|
| "什么是RAG？" | LLM 分类 / 关键词"什么是" | 关键词 + 向量混合搜索 |
| "解释一下注意力机制" | LLM 分类 / 默认兜底 | 纯向量搜索（语义相似度） |
| "A和B有什么区别" | LLM 分类 / 关键词"区别" | 混合搜索 + RRF 融合 |
| "分析RAG的优缺点" | LLM 分类 / 关键词"分析" | 向量搜索 + 知识图谱拓展 |
| "今天比特币价格" | LLM 分类 / 关键词"latest" | 不检索，返回空等 web_search |
