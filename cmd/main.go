// VolSeek-Agent — 一个具有规划、执行和反思能力的智能 RAG Agent。
//
// 主要特性：
//   - Plan-then-Execute：Agent 先制定计划再执行，而非传统 ReAct 的边想边做
//   - GraphRAG：知识图谱增强检索，理解实体间关系
//   - Self-Reflection：答案生成后自我评审，确保质量
//   - Adaptive Routing：查询意图路由，自动选择最优检索策略
//   - Streaming SSE：流式输出，实时展示 Agent 的思考过程
//
// 使用方法：
//   1. 复制 .env.example 为 .env 并填入你的 API Key
//   2. go run cmd/main.go "你的问题"
//   3. 或者编译后运行：go build -o volseek cmd/main.go && ./volseek "你的问题"
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/agent"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/config"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/llm"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/rag"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/tools"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

func main() {
	// ========================================================================
	// 加载配置
	// ========================================================================
	if err := config.Load(); err != nil {
		log.Printf("Warning: config load: %v", err)
	}

	apiKey := config.GetEnv("LLM_API_KEY", "")
	baseURL := config.GetEnv("LLM_BASE_URL", "https://api.openai.com/v1")
	model := config.GetEnv("LLM_MODEL", "gpt-4o-mini")

	// Embedding 配置（可独立于聊天 API，如 DeepSeek 聊天 + SiliconFlow 向量化）
	embedBaseURL := config.GetEnv("EMBEDDING_BASE_URL", baseURL)
	embedModel := config.GetEnv("EMBEDDING_MODEL", "text-embedding-ada-002")
	embedAPIKey := config.GetEnv("EMBEDDING_API_KEY", apiKey) // 没配则复用聊天密钥

	if apiKey == "" {
		fmt.Println("⚠️  LLM_API_KEY 未设置！")
		fmt.Println("请复制 .env.example 为 .env 并填入你的 API Key")
		fmt.Println("或通过环境变量设置: set LLM_API_KEY=sk-your-key")
		fmt.Println()
	}

	// ========================================================================
	// 初始化 LLM 客户端
	// ========================================================================
	fmt.Print("🔄 Initializing LLM client... ")
	llmClient := llm.NewClient(apiKey, baseURL, model, embedBaseURL, embedModel, embedAPIKey)
	fmt.Println("✅")

	// ========================================================================
	// 初始化 RAG 引擎
	// ========================================================================
	fmt.Print("🔄 Initializing RAG engine... ")

	// 创建分块器（每块 500 字符，重叠 50）
	chunker := rag.NewChunker(&types.ChunkConfig{Size: 500, Overlap: 50})

	// 创建向量存储
	vectorStore := rag.NewVectorStore()

	// 创建知识图谱存储
	graphStore := rag.NewGraphStore()

	// 创建查询意图路由器
	queryRouter := rag.NewQueryRouter(llmClient)

	// 创建多策略检索器
	retriever := rag.NewRetriever(vectorStore, llmClient, graphStore, 5)

	fmt.Println("✅")

	// ========================================================================
	// 索引示例文档
	// ========================================================================
	indexSampleDocuments(chunker, llmClient, vectorStore, graphStore)

	// ========================================================================
	// 初始化工具注册中心
	// ========================================================================
	fmt.Print("🔄 Initializing tools... ")
	registry := tools.NewRegistry()

	// 知识搜索工具（使用检索器）
	registry.Register(tools.NewKnowledgeSearchTool(func(ctx context.Context, query string) ([]*types.SearchResult, error) {
		// 先用查询路由器分析意图
		intent, err := queryRouter.Analyze(ctx, query)
		if err != nil {
			intent = &types.QueryIntent{Type: types.QueryConceptual}
		}
		return retriever.Retrieve(ctx, query, intent)
	}))

	// 知识图谱搜索工具
	if graphEntities, _ := graphStore.Stats(); graphEntities > 0 {
		registry.Register(tools.NewGraphSearchTool(func(ctx context.Context, entity string) ([]*types.SearchResult, error) {
			entities := graphStore.FindEntities(entity)
			if len(entities) == 0 {
				return nil, nil
			}
			var results []*types.SearchResult
			for _, e := range entities {
				neighbors := graphStore.GetNeighbors(e.ID, 1)
				for _, n := range neighbors {
					results = append(results, &types.SearchResult{
						Chunk: &types.Chunk{
							ID:       n.ID,
							Content:  n.Context,
							DocTitle: n.Name,
						},
						Score:  0.8,
						Method: "graph",
					})
				}
			}
			return results, nil
		}))
	}

	// 计算器工具
	registry.Register(tools.NewCalculatorTool())

	// 最终答案工具（必须注册）
	registry.Register(tools.NewFinalAnswerTool())

	fmt.Println("✅")

	// ========================================================================
	// 初始化 Agent 引擎
	// ========================================================================
	fmt.Print("🔄 Initializing Agent engine... ")

	agentCfg := &types.AgentConfig{
		Temperature:       0.7,
		MaxPlanningRounds: 10,
		EnableReflection:  true,  // 启用自我反思
		EnableGraphRAG:    true,  // 启用知识图谱
		ParallelToolCalls: true,  // 启用并行工具调用
		MaxContextTokens:  64000, // 上下文窗口
	}

	volseek := agent.NewAgentEngine(agentCfg, llmClient, registry, queryRouter, retriever)

	fmt.Println("✅")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("🤖 VolSeek-Agent 已就绪！")
	fmt.Println(strings.Repeat("=", 60))

	// ========================================================================
	// 交互模式
	// ========================================================================
	// 从命令行参数或交互式输入获取查询
	query := strings.Join(os.Args[1:], " ")

	if query == "" {
		// 交互模式
		fmt.Println("\n💡 输入你的问题（输入 'exit' 退出, 'stats' 查看状态）:")
		scanner := bufio.NewScanner(os.Stdin)

		for {
			fmt.Print("\n❓ ")
			scanner.Scan()
			query = strings.TrimSpace(scanner.Text())

			if query == "" {
				continue
			}
			if query == "exit" || query == "quit" {
				fmt.Println("👋 再见！")
				return
			}
			if query == "stats" {
				showStats(vectorStore, graphStore)
				continue
			}

			runQuery(context.Background(), volseek, query)
		}
	} else {
		// 直接执行
		runQuery(context.Background(), volseek, query)
	}
}

// runQuery 执行一次查询并显示结果。
func runQuery(ctx context.Context, volseek *agent.AgentEngine, query string) {
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("🔍 Query: %s\n", query)
	fmt.Println(strings.Repeat("-", 60))

	startTime := time.Now()

	// 获取流式事件
	eventCh, err := volseek.Execute(ctx, query)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return
	}

	// 处理流式事件
	var confidence float64

	for event := range eventCh {
		switch event.Type {
		case types.EventPlan:
			fmt.Printf("\n📋 %s\n", event.Content)
		case types.EventThink:
			fmt.Printf("\n🤔 %s\n", event.Content)
		case types.EventToolCall:
			fmt.Printf("\n🔧 %s\n", event.Content)
		case types.EventToolResult:
			fmt.Printf("   %s\n", event.Content)
		case types.EventReflect:
			fmt.Printf("\n🔍 %s\n", event.Content)
		case types.EventAnswer:
			fmt.Printf("\n📝 %s\n", event.Content)
		case types.EventError:
			fmt.Printf("\n❌ %s\n", event.Content)
		case types.EventDone:
			if data, ok := event.Data.(map[string]interface{}); ok {
				if c, ok := data["confidence"].(float64); ok {
					confidence = c
				}
			}
		}
	}

	elapsed := time.Since(startTime)

	// 显示执行摘要
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("⏱️  Time: %v\n", elapsed.Round(time.Millisecond))
	if confidence > 0 {
		fmt.Printf("🎯 Confidence: %.0f%%\n", confidence*100)
	}
	fmt.Println(strings.Repeat("=", 60))
}

// indexSampleDocuments 索引示例文档到知识库。
func indexSampleDocuments(chunker *rag.Chunker, llmClient *llm.Client, vs *rag.VectorStore, gs *rag.GraphStore) {
	fmt.Print("🔄 Indexing sample documents... ")

	docs := map[string]string{
		"RAG 技术介绍": `RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与文本生成的人工智能技术。
RAG 的核心思想是：在 LLM 生成答案之前，先从外部知识库中检索相关信息，然后将检索到的内容作为上下文提供给 LLM。
这种方法有效解决了传统 LLM 的三大问题：
1. 知识截止日期：LLM 的训练数据只到某个时间点，之后的信息无法获知
2. 幻觉问题：LLM 可能编造看起来合理但实际上错误的信息
3. 领域知识不足：通用 LLM 缺乏特定领域的专业知识

RAG 的典型工作流程包括三个步骤：
第一步（索引）：将文档分块、向量化，存入向量数据库
第二步（检索）：将用户问题向量化，在数据库中搜索最相似的文档块
第三步（生成）：将检索到的文档块作为上下文，让 LLM 生成答案

RAG 广泛应用于智能客服、企业知识库问答、法律文档分析、医疗辅助诊断等领域。
GraphRAG 是 RAG 的进阶版本，通过构建知识图谱来理解实体间的关系，提供更深入的分析能力。`,

		"Go 语言入门": `Go（又称 Golang）是 Google 开发的开源编程语言，于 2009 年正式对外发布。
Go 语言的设计者是 Robert Griesemer、Rob Pike 和 Ken Thompson（Unix 和 C 语言的共同创造者）。
Go 的三大设计目标是：简洁的语法、高效的编译、强大的并发支持。

Go 语言的主要特性包括：
- 静态类型和类型推断
- 垃圾回收机制
- 内置并发原语（goroutine 和 channel）
- 快速编译
- 标准库丰富
- 跨平台支持

goroutine 是 Go 语言最著名的特性，它是一种轻量级线程，创建成本极低（仅几 KB）。
通过 go 关键字可以启动一个 goroutine：go func() { ... }()
channel 是 goroutine 之间的通信机制，遵循"CSP"（Communicating Sequential Processes）模型。`,

		"VolSeek-Agent 设计文档": `VolSeek-Agent 是一个具有规划、执行和反思能力的智能 RAG Agent 框架。
其核心架构分为三个主要阶段：

1. 规划阶段（Plan）：Agent 收到用户问题后，首先通过意图路由器分析问题类型（事实性、概念性、比较性等），然后由 LLM 生成结构化的执行计划。
   执行计划包含目标重述、子目标列表和所需工具的判断。

2. 执行阶段（Execute）：Agent 按计划逐步执行，每步调用 LLM + 工具。
   核心工具包括：knowledge_search（语义搜索）、graph_search（图谱搜索）、web_search（网页搜索）、calculator（计算器）。
   支持并行工具调用，当多个工具相互独立时并发执行以提高效率。

3. 反思阶段（Reflect）：Agent 生成答案后对其质量进行自我评审。
   评审维度包括：准确性（是否有事实错误）、完整性（是否回答所有问题点）、清晰度（表达是否易懂）。
   如果评审发现问题，Agent 会自动修正答案。

VolSeek-Agent 的技术栈：Go 语言、OpenAI/DeepSeek API、向量检索、知识图谱。`,
	}

	// 索引每个文档
	var allDocs []*types.Chunk
	for title, content := range docs {
		chunks := chunker.Chunk(content, title)
		allDocs = append(allDocs, chunks...)
	}

	// 批量向量化
	texts := make([]string, len(allDocs))
	for i, chunk := range allDocs {
		texts[i] = chunk.Content
	}

	if len(texts) > 0 {
		embeddings, err := llmClient.EmbedBatch(context.Background(), texts)
		if err != nil {
			log.Printf("Warning: embedding failed: %v (using keyword search only)", err)
		} else {
			for i, emb := range embeddings {
				if emb != nil {
					allDocs[i].Embedding = emb
				}
			}
		}
	}

	// 存入向量存储
	vs.AddBatch(allDocs)

	// 构建知识图谱
	gs.BuildFromChunks(allDocs, llmClient)

	fmt.Printf("✅ (%d chunks, %d entities, %d relations)\n", len(allDocs), func() int { e, _ := gs.Stats(); return e }(), func() int { _, r := gs.Stats(); return r }())
}

// showStats 显示系统状态。
func showStats(vs *rag.VectorStore, gs *rag.GraphStore) {
	entities, relations := gs.Stats()
	fmt.Println(strings.Repeat("-", 40))
	fmt.Println("📊 System Status:")
	fmt.Printf("  Chunks in vector store: %d\n", vs.Len())
	fmt.Printf("  Graph entities: %d\n", entities)
	fmt.Printf("  Graph relations: %d\n", relations)
	fmt.Println(strings.Repeat("-", 40))
}
