# 🤖 VolSeek-Agent

> **Plan-then-Execute 智能 RAG Agent 框架** — HTTP API 服务 + Web 前端  
> 从零构建，支持向量检索、自我反思和多工具调用的智能 Agent

[![Go Version](https://img.shields.io/badge/Go-1.25+-00ADD8)](https://go.dev)
[![Gin](https://img.shields.io/badge/Gin-1.12-008ECF)](https://gin-gonic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](#)

---

## ✨ 核心特性

| 特性                            | 说明 |
|-------------------------------|------|
| **🧠 Plan-then-Execute**      | Agent 先制定结构化计划再执行，告别传统 ReAct 的盲目循环 |
| **🔎 自我反思 (Self-Reflection)** | 生成答案后自动评审质量，修正错误，输出置信度评分 |
| **🎯 自适应查询路由**                | LLM 分析查询意图 (factual/conceptual/comparative/analytical)，自动选择最优检索策略 |
| **💾 双级记忆系统**                 | 工作记忆 + 情节记忆（环形缓冲区 + LLM 摘要压缩） |
| **📁 知识库管理**                  | 上传/删除/重新索引 `.txt` / `.md` 文件，自动分块、嵌入 |
| **🔌 5 个内置工具**                | `knowledge_search` ·  `web_search` · `calculator` · `memory_recall` · `final_answer` |
| **🌐 HTTP API + SSE 流式**      | 基于 Gin 的 RESTful API，支持 Server-Sent Events 实时推送 |
| **🖥️ Web 前端**                | 原生 SPA，支持流式渲染、Markdown 显示、文件上传、快速/深度模式切换 |
| **🔁 弹性向量存储**                 | 内存向量存储（默认） / Qdrant 专业向量数据库（可选，零配置切换） |
| **🔧 可选的 Rerank 重排**          | 交叉编码器二次精排，进一步提升检索排序质量 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            VolSeek-Agent                                    │
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐   ┌─────────────┐  │
│  │   LLM    │   │   RAG    │   │       Agent          │   │    Tools    │  │
│  │  Client  │   │  Engine  │   │  ┌──────┬──────┬──┐  │   │  Registry   │  │
│  │ Chat/Embed│◀─│ Chunker │   │  │ Plan │Exec  │Ref│  │   │             │  │
│  │ Stream+Retry│  │Retriever│   │  │      │ute   │ect│  │   │ knowledge  │  │
│  │          │   │  Router   │   │  └──────┴──────┴──┘  │   │ web        │  │
│  └──────────┘   │ Reranker  │   └──────────────────────┘   │ calculator │  │
│                 └───────────┘                              │ memory     │  │
│                                                            │ final_answer│  │
│                                                            └─────────────┘  │
│           ┌──────────────────┐       ┌──────────────┐                     │
│           │  Vector Store    │       │    Memory    │                     │
│           │ (Memory / Qdrant)│       │    Buffer    │                     │
│           └──────────────────┘       └──────────────┘                     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         cmd/main.go                                  │  │
│  │           Gin HTTP Server (:8080) + Graceful Shutdown                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           │                                                │
│              ┌────────────┴────────────┐                                   │
│              ▼                         ▼                                   │
│   ┌──────────────────┐     ┌──────────────────┐                            │
│   │   Web Frontend   │     │  External APIs   │                            │
│   │   (SPA / SSE)    │     │ (curl / Postman)  │                            │
│   └──────────────────┘     └──────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent 执行流程

```
用户提问 ──▶  1. Plan（制定计划）
                 ├── QueryRouter 分析意图
                 └── LLM 生成结构化 plan（goal + sub_goals）
             2. Execute（循环执行）
                 ├── Think（LLM 思考）
                 ├── Act（调用工具，支持并行）
                 └── Observe（收集结果）
             3. Reflect（自我反思） ──▶ 评审质量，修正答案
                                    ──▶ 输出最终答案 + 置信度
```

---

## 🛠️ 技术栈

| 层 | 技术 |
|----|------|
| **语言** | Go 1.25 |
| **Web 框架** | [Gin](https://gin-gonic.com) v1.12 |
| **LLM 客户端** | [go-openai](https://github.com/sashabaranov/go-openai) v1.41（兼容 OpenAI / DeepSeek / SiliconFlow / Ollama） |
| **向量数据库** | 内存向量存储（默认） / [Qdrant](https://qdrant.tech) v1.18（可选） |
| **Token 计数** | [tiktoken-go](https://github.com/pkoukk/tiktoken-go) v0.1（用于 memory benchmark） |
| **配置** | [godotenv](https://github.com/joho/godotenv) v1.5 |
| **前端** | 原生 HTML/CSS/JS · [marked.js](https://marked.js.org) · [highlight.js](https://highlightjs.org) |

---

## 🚀 快速启动

### 前置条件

- Go 1.21+（源码运行）**或** [预编译二进制](https://github.com/qingutaoo-design/VolSeek-Agent/releases)
- 一个兼容 OpenAI API 的服务商密钥

### 1. 克隆项目

```bash
git clone https://github.com/qingutaoo-design/VolSeek-Agent.git
cd VolSeek-Agent
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API 密钥。支持三种方案：

#### 🥇 OpenAI 一站式

```ini
LLM_API_KEY=sk-**************
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

#### 🥈 DeepSeek 聊天 + SiliconFlow 向量化（中文优化，推荐）

```ini
LLM_API_KEY=sk-**************
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

EMBEDDING_BASE_URL=https://api.siliconflow.com/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

#### 🥉 Ollama 全本地（免费）

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

```ini
LLM_API_KEY=sk-placeholder
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3

EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
```

### 3. 运行

```bash
# 源码运行（HTTP 服务，端口 8080）
go run cmd/main.go

# 编译后运行
go build -o volseek cmd/main.go
./volseek        # Linux / macOS
.\volseek.exe    # Windows
```

### 4. 打开浏览器

访问 **[http://localhost:8080](http://localhost:8080)**，即可开始对话。

项目内置 3 篇示例文档（RAG 技术介绍、Go 语言入门、VolSeek 设计文档），启动后自动索引，可直接提问：

```
❓ 什么是 RAG？
❓ 什么是 goroutine？
❓ 分析 RAG 技术的优缺点
```

---

## 📚 API 文档

所有 API 端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | Web 前端 SPA 页面 |
| `GET` | `/api/health` | 健康检查（返回 chunks 数量、服务状态） |
| `GET` | `/api/stats` | 系统统计（chunks / entities / relations 数量） |
| `POST` | `/api/query` | **流式查询** — SSE 事件流，实时推送思考过程 |
| `POST` | `/api/query/sync` | **同步查询** — 阻塞等待完整 JSON 返回 |
| `POST` | `/api/chat` | **会话查询** — 带 `session_id`，可维持多轮对话记忆 |
| `POST` | `/api/knowledge/upload` | 上传文件到知识库（multipart/form-data） |
| `GET` | `/api/knowledge/files` | 列出知识库所有文件 |
| `DELETE` | `/api/knowledge/files/:uuid` | 删除指定知识库文件 |
| `POST` | `/api/knowledge/import` | 从服务器本地路径导入文件 |
| `POST` | `/api/knowledge/reindex` | 重新索引所有知识库文件 |

### 流式查询示例

```bash
curl -N -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"什么是 RAG？"}'
```

返回 SSE 事件流：

```
event: message
data: {"type":"plan","content":"Analyzing your question..."}

event: message
data: {"type":"tool_call","content":"Searching knowledge base..."}

event: message
data: {"type":"answer","content":"RAG（Retrieval-Augmented Generation）是..."}

event: message
data: {"type":"done","content":"","data":{"confidence":0.92}}
```

### 同步查询示例

```bash
curl -X POST http://localhost:8080/api/query/sync \
  -H "Content-Type: application/json" \
  -d '{"query":"什么是 RAG？"}'
```

```json
{
  "answer": "RAG（Retrieval-Augmented Generation）是一种结合...",
  "confidence": 0.95
}
```

### 会话查询示例

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"刚才我们聊了什么？","session_id":"my-session-001"}'
```

```json
{
  "answer": "我们刚才讨论了 RAG 的基本概念...",
  "confidence": 0.88,
  "session_id": "my-session-001"
}
```

### 知识库管理示例

```bash
# 上传文件
curl -X POST http://localhost:8080/api/knowledge/upload \
  -F "file=@/path/to/document.md"

# 列出文件
curl http://localhost:8080/api/knowledge/files

# 删除文件
curl -X DELETE http://localhost:8080/api/knowledge/files/<uuid>

# 重新索引
curl -X POST http://localhost:8080/api/knowledge/reindex
```

---

## 📁 项目结构

```
VolSeek-Agent/
├── cmd/
│   └── main.go                  # 入口：Gin HTTP 服务 + 优雅关闭
├── internal/
│   ├── types/types.go           # 核心类型定义（零外部依赖）
│   ├── config/config.go         # 环境变量加载工具
│   ├── llm/client.go            # LLM 客户端（Chat/Embed/Stream + 重试）
│   ├── rag/
│   │   ├── rag.go               # RAG 引擎：Chunker + Retriever + QueryRouter
│   │   ├── qdrant_store.go      # Qdrant 向量数据库适配器
│   │   └── reranker.go          # 交叉编码器重排器
│   ├── agent/engine.go          # Agent 引擎：Plan → Execute → Reflect
│   ├── tools/tools.go           # 工具注册中心 + 6 个内置工具
│   ├── memory/
│   │   ├── memory.go            # 记忆接口定义
│   │   └── buffer.go            # 环形缓冲区记忆（LLM 摘要压缩）
│   ├── knowledge/manager.go     # 知识库文件管理（上传/删除/索引）
│   └── initapp/
│       ├── init.go              # 初始化编排（LLM→RAG→Tools→Agent→KB）
│       ├── index.go             # 示例文档索引
│       └── router.go            # Gin 路由注册
├── frontend/
│   ├── index.html               # SPA 前端页面
│   ├── css/style.css            # 样式表
│   └── js/app.js                # 前端逻辑（流式 SSE + 文件上传 + 模式切换）
├── knowledge_base/              # 知识库文件存储目录
│   └── kb_meta.json             # 文件元数据索引
├── .env.example                 # 环境变量模板
├── go.mod / go.sum              # Go 依赖
└── README.md                    # 本文件
```

---

## ⚙️ 配置参考

所有配置项通过 `.env` 文件或环境变量设置：

| 变量 | 默认值 | 必填 | 说明 |
|------|--------|------|------|
| `LLM_API_KEY` | — | ✅ | 聊天 API 密钥 |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | ✅ | 聊天 API 端点 |
| `LLM_MODEL` | `gpt-4o-mini` | ✅ | 聊天模型名称 |
| `EMBEDDING_BASE_URL` | 同 `LLM_BASE_URL` | — | 向量化 API 端点（不同服务商时单独设置） |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | — | 向量化模型名称 |
| `EMBEDDING_API_KEY` | 同 `LLM_API_KEY` | — | 向量化 API 密钥（与聊天不同时设置） |
| `SERP_API_KEY` | — | — | 网页搜索 API 密钥（可选） |
| `RERANK_API_KEY` | — | — | 重排 API 密钥（可选） |
| `RERANK_BASE_URL` | — | — | 重排 API 端点 |
| `RERANK_MODEL` | — | — | 重排模型名称（如 `BAAI/bge-reranker-v2-m3`） |
| `QDRANT_HOST` | — | — | Qdrant 服务地址（留空使用内存向量存储） |
| `QDRANT_COLLECTION` | `volseek` | — | Qdrant 集合名称 |
| `QDRANT_DIMENSION` | `1024` | — | 向量维数 |
| `KB_DIR` | `./knowledge_base` | — | 知识库文件存储目录 |

### 推荐 Embedding 模型

| 服务商 | 模型 | 特点 |
|-------|------|------|
| OpenAI | `text-embedding-ada-002` | 通用，1536 维 |
| OpenAI | `text-embedding-3-small` | 性价比更高 |
| SiliconFlow（免费） | `BAAI/bge-large-zh-v1.5` | 中文优化，推荐 |
| SiliconFlow（免费） | `Qwen/Qwen3-Embedding-0.6B` | 极致低价 |
| Ollama 本地 | `nomic-embed-text` | 完全免费 |

---

## 🧠 Agent 核心设计

### Plan-then-Execute

与传统的 ReAct（边想边做）模式不同，VolSeek 采用 **先规划后执行**：

```
传统 ReAct:   思考 → 行动 → 观察 → 思考 → 行动 → ...（容易发散）
VolSeek:      [规划] 制定完整计划 → [执行] 按计划调用工具 → [反思] 评审答案
```

### 自适应查询路由

系统自动识别查询意图并选择最优策略：

| 查询类型 | 示例 | 检索策略 |
|---------|------|---------|
| Factual（事实） | "什么是 RAG？" | 向量 + 关键词混合搜索 |
| Conceptual（概念） | "解释一下注意力机制" | 纯向量搜索 |
| Comparative（比较） | "A 和 B 有什么区别" | 混合搜索 + RRF 融合排序 |
| Analytical（分析） | "分析 RAG 的优缺点" | 向量搜索 + 关键词混合搜索 |
| Recent（实时） | "今天比特币价格" | 不检索，等待 web_search |

### 自我反思

生成答案后，Agent 从三个维度评审质量：
- **准确性**：事实是否有误
- **完整性**：是否全面回答问题
- **清晰度**：是否清晰易懂

评审通过后输出改进后的答案和置信度评分。

---

## 🔧 开发

### 常用命令

```bash
# 运行服务
go run cmd/main.go

# 编译
go build -o volseek cmd/main.go

# 分块策略对比测试
go test ./internal/rag/ -v -run TestCompareStrats

# 记忆系统基准测试
go test ./internal/memory/ -bench=.
```

### 扩展工具

实现自定义工具只需实现 `Tool` 接口：

```go
type Tool interface {
    Name() string
    Description() string
    Parameters() json.RawMessage
    Execute(ctx context.Context, args json.RawMessage) *types.ToolResult
}
```

然后在 `internal/initapp/init.go` 中注册：

```go
registry.Register(yourNewTool)
```

### 启用 Qdrant

1. 启动 Qdrant：`docker run -p 6333:6333 qdrant/qdrant`
2. 在 `.env` 中添加 `QDRANT_HOST=localhost:6333`

### 启用 Rerank 重排

在 `.env` 中配置：

```ini
RERANK_API_KEY=sk-**************
RERANK_BASE_URL=https://api.siliconflow.com/v1
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

---

## ❓ 常见问题

**启动时报 "embedding failed"** → 你的聊天 API 不支持向量化（如 DeepSeek），需单独配置 `EMBEDDING_BASE_URL` 和 `EMBEDDING_MODEL`。

**"401 Unauthorized"** → API 密钥与端点不匹配。检查 `LLM_API_KEY` 是否对应 `LLM_BASE_URL` 指向的服务商。

**"模型不存在" / "404"** → 模型名称在当前服务商不可用。参考服务商官方文档确认模型名。

**回答质量差** → 中文文档建议使用中文优化模型（如 `BAAI/bge-large-zh-v1.5`）。

---

## 📄 许可

[MIT License](LICENSE)
