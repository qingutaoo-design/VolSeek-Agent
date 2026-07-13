// VolSeek-Agent — 一个具有规划、执行和反思能力的智能 RAG Agent。
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/agent"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/config"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/llm"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/rag"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/tools"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

func main() {
	if err := config.Load(); err != nil {
		log.Printf("Warning: config load: %v", err)
	}

	volseek, store, graphStore, _ := initAgent(context.Background())

	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("🤖 VolSeek-Agent 已就绪！")
	fmt.Println(strings.Repeat("=", 60))

	router := newRouter(volseek, store, graphStore)
	srv := &http.Server{Addr: ":8080", Handler: router}

	go func() {
		log.Printf("🌐 API server on http://localhost%s", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("API server failed: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	log.Printf("🛑 Received %v, shutting down...", sig)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	srv.Shutdown(ctx)
	log.Println("👋 Exited gracefully")
}

func newRouter(volseek *agent.AgentEngine, store rag.Store, graphStore *rag.GraphStore) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type")
		if c.Request.Method == "OPTIONS" { c.AbortWithStatus(204); return }
		c.Next()
	})

	r.GET("/api/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok", "chunks": store.Len(), "time": time.Now().Format(time.RFC3339)})
	})
	r.GET("/api/stats", func(c *gin.Context) {
		e, r2 := graphStore.Stats()
		c.JSON(200, gin.H{"chunks": store.Len(), "entities": e, "relations": r2})
	})
	r.POST("/api/query", func(c *gin.Context) {
		var req struct{ Query string `json:"query" binding:"required"` }
		if err := c.ShouldBindJSON(&req); err != nil { c.JSON(400, gin.H{"error": "missing query"}); return }
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		eventCh, err := volseek.Execute(ctx, req.Query)
		if err != nil { c.SSEvent("error", gin.H{"message": err.Error()}); c.Writer.Flush(); return }
		for event := range eventCh {
			c.SSEvent("message", gin.H{"type": event.Type, "content": event.Content, "done": event.Done})
			c.Writer.Flush()
			if event.Done { break }
		}
	})
	r.POST("/api/query/sync", func(c *gin.Context) {
		var req struct{ Query string `json:"query" binding:"required"` }
		if err := c.ShouldBindJSON(&req); err != nil { c.JSON(400, gin.H{"error": "missing query"}); return }
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		eventCh, err := volseek.Execute(ctx, req.Query)
		if err != nil { c.JSON(500, gin.H{"error": err.Error()}); return }
		var answer string; var conf float64; var steps []string
		for event := range eventCh {
			switch event.Type {
			case types.EventAnswer: answer += event.Content
			case types.EventToolCall: steps = append(steps, event.Content)
			case types.EventDone:
				if d, ok := event.Data.(map[string]interface{}); ok {
					if c, ok := d["confidence"].(float64); ok { conf = c }
				}
			}
		}
		c.JSON(200, gin.H{"answer": answer, "confidence": conf, "steps": steps})
	})
	return r
}

func initAgent(ctx context.Context) (*agent.AgentEngine, rag.Store, *rag.GraphStore, *llm.Client) {
	apiKey := config.GetEnv("LLM_API_KEY", "")
	baseURL := config.GetEnv("LLM_BASE_URL", "https://api.openai.com/v1")
	model := config.GetEnv("LLM_MODEL", "gpt-4o-mini")
	embedBaseURL := config.GetEnv("EMBEDDING_BASE_URL", baseURL)
	embedModel := config.GetEnv("EMBEDDING_MODEL", "text-embedding-ada-002")
	embedAPIKey := config.GetEnv("EMBEDDING_API_KEY", apiKey)

	fmt.Print("🔄 Initializing LLM client... ")
	llmClient := llm.NewClient(apiKey, baseURL, model, embedBaseURL, embedModel, embedAPIKey)
	fmt.Println("✅")

	fmt.Print("🔄 Initializing RAG engine... ")
	chunker := rag.NewChunker(&types.ChunkConfig{Size: 500, Overlap: 80, MinSize: 125})
	qdrantHost := config.GetEnv("QDRANT_HOST", "")
	var store rag.Store
	if qdrantHost != "" {
		qdrantCollection := config.GetEnv("QDRANT_COLLECTION", "volseek")
		qdrantDim := config.GetEnvInt("QDRANT_DIMENSION", 1024)
		qs, err := rag.NewQdrantStore(ctx, qdrantHost, qdrantCollection, uint64(qdrantDim))
		if err != nil {
			log.Printf("Warning: Qdrant init failed (%v), falling back to in-memory store", err)
			store = rag.NewVectorStore()
		} else {
			store = qs; fmt.Println("✅ (Qdrant)"); fmt.Print("🔄 Loading into Qdrant... ")
		}
	} else { store = rag.NewVectorStore() }

	graphStore := rag.NewGraphStore()
	queryRouter := rag.NewQueryRouter(llmClient)
	retriever := rag.NewRetriever(store, llmClient, graphStore, 5)
	fmt.Println("✅")

	fmt.Print("🔄 Initializing tools... ")
	registry := tools.NewRegistry()
	registry.Register(tools.NewKnowledgeSearchTool(func(ctx context.Context, q string) ([]*types.SearchResult, error) {
		intent, err := queryRouter.Analyze(ctx, q)
		if err != nil { intent = &types.QueryIntent{Type: types.QueryConceptual} }
		return retriever.Retrieve(ctx, q, intent)
	}))
	if graphEntities, _ := graphStore.Stats(); graphEntities > 0 {
		registry.Register(tools.NewGraphSearchTool(func(ctx context.Context, entity string) ([]*types.SearchResult, error) {
			entities := graphStore.FindEntities(entity)
			if len(entities) == 0 { return nil, nil }
			var results []*types.SearchResult
			for _, e := range entities {
				for _, n := range graphStore.GetNeighbors(e.ID, 1) {
					results = append(results, &types.SearchResult{
						Chunk: &types.Chunk{ID: n.ID, Content: n.Context, DocTitle: n.Name},
						Score: 0.8, Method: "graph",
					})
				}
			}
			return results, nil
		}))
	}
	registry.Register(tools.NewCalculatorTool())
	registry.Register(tools.NewFinalAnswerTool())
	fmt.Println("✅")

	fmt.Print("🔄 Initializing Agent engine... ")
	agentCfg := &types.AgentConfig{
		Temperature: 0.7, MaxPlanningRounds: 10,
		EnableReflection: true, EnableGraphRAG: true,
		ParallelToolCalls: true, MaxContextTokens: 64000,
	}
	volseek := agent.NewAgentEngine(agentCfg, llmClient, registry, queryRouter, retriever)
	fmt.Println("✅")

	indexSampleDocuments(chunker, llmClient, store, graphStore)
	return volseek, store, graphStore, llmClient
}
// docEntry 表示一篇待索引的文档。
type docEntry struct {
	uuid    string
	title   string
	content string
}

func checkDocHash(qs *rag.QdrantStore, docUUID, expectedHash string) bool {
	points, err := qs.ScrollByFilter(context.Background(), map[string]string{"doc_uuid": docUUID}, 1)
	if err != nil || len(points) == 0 {
		return false
	}
	if v, ok := points[0].Payload["doc_hash"]; ok {
		return v.GetStringValue() == expectedHash
	}
	return false
}

func indexSampleDocuments(chunker *rag.Chunker, llmClient *llm.Client, store rag.Store, gs *rag.GraphStore) {
	fmt.Print("🔄 Indexing sample documents... ")

	docs := []docEntry{
		{"doc-rag-intro", "RAG 技术介绍", `RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与文本生成的人工智能技术。
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
GraphRAG 是 RAG 的进阶版本，通过构建知识图谱来理解实体间的关系，提供更深入的分析能力。`},
		{"doc-go-intro", "Go 语言入门", `Go（又称 Golang）是 Google 开发的开源编程语言，于 2009 年正式对外发布。
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
channel 是 goroutine 之间的通信机制，遵循"CSP"（Communicating Sequential Processes）模型。`},
		{"doc-volseek", "VolSeek-Agent 设计文档", `VolSeek-Agent 是一个具有规划、执行和反思能力的智能 RAG Agent 框架。
其核心架构分为三个主要阶段：

1. 规划阶段（Plan）：Agent 收到用户问题后，首先通过意图路由器分析问题类型（事实性、概念性、比较性等），然后由 LLM 生成结构化的执行计划。
   执行计划包含目标重述、子目标列表和所需工具的判断。

2. 执行阶段（Execute）：Agent 按计划逐步执行，每步调用 LLM + 工具。
   核心工具包括：knowledge_search（语义搜索）、graph_search（图谱搜索）、web_search（网页搜索）、calculator（计算器）。
   支持并行工具调用，当多个工具相互独立时并发执行以提高效率。

3. 反思阶段（Reflect）：Agent 生成答案后对其质量进行自我评审。
   评审维度包括：准确性（是否有事实错误）、完整性（是否回答所有问题点）、清晰度（表达是否易懂）。
   如果评审发现问题，Agent 会自动修正答案。

VolSeek-Agent 的技术栈：Go 语言、OpenAI/DeepSeek API、向量检索、知识图谱。`},
		{"doc-planexecute-replan", "plan-execute-replan", `Plan-Execute-Replan 是什么？
先给定义：Plan-Execute-Replan 是 Agent 的 结构化任务执行模式

核心是 先规划执行步骤，再按步骤行动，随时校准方向， 通过 规划→执行→评估→调整 ，让 Agent 像项目经理一样拆解复杂任务、稳步推进，还能应对突发变化。

大白话解释：装修房子的全流程
想象你要装修一套新房，完全没经验的话，你会怎么避免手忙脚乱？

Plan（规划） ：先找设计师出详细方案，拆改哪里、水电怎么走、用什么材料、分几个阶段施工（比如拆旧→水电→泥工→木工→油漆），形成清晰的步骤清单。

Execute（执行） ：施工队按计划开工，先拆旧、再布水电，一步一步推进当前阶段的任务。

Replan（重规划） ：施工中发现原计划的承重墙不能拆，则设计师重新调整布局（比如把书房门改到另一侧），更新计划后继续施工，直到最终完工。

Plan-Execute-Replan 就是让 Agent 模仿这个过程：遇到复杂任务不盲目动手，而是先设计方案，再按图施工，遇到问题时灵活调整方案，确保最终达成目标。

技术场景例子：运维Agent的故障排查实战
假设某电商平台服务器凌晨突发 CPU使用率100% 告警，需要运维Agent自动排查根因。这是典型的多步骤运维任务，正好适合 Plan-Execute-Replan 模式：

Plan：拆解排查步骤
Planner（规划器）接到 排查CPU突增根因 的目标后，结合运维经验生成结构化计划：

步骤1：调用日志工具，查询服务器近1小时error/warn级别日志（重点看进程崩溃、资源争抢记录）  
步骤2：调用监控工具，获取CPU使用率突增时段的进程占用排行（定位高耗CPU进程）  
步骤3：调用历史工单手册，检索该进程过往CPU异常的处理方案（匹配已知问题）  
Execute：执行第一步查日志
Executor（执行器）按计划启动第一步，调用日志工具，参数为服务器IP=10.0.1.5，时间范围=近1小时，日志级别=error/warn，返回结果：

日志中未发现error/warn记录，仅存在大量info级别的定时任务执行成功日志（无异常线索）  
Replan：评估结果并调整计划
Replanner（重规划器）分析执行结果：日志无异常，说明问题可能不在应用错误，需优先定位高耗CPU进程，于是调整计划顺序：

【更新后计划】  
步骤1：调用监控工具，获取CPU使用率突增时段（02:00-02:10）的进程占用排行（优先定位异常进程）  
步骤2：调用日志工具，查询步骤1中高耗CPU进程的近1小时详细日志（针对性排查该进程行为）  
步骤3：调用历史工单手册，检索该进程过往CPU异常的处理方案  
继续执行与动态调整
Executor 执行更新后的步骤1，调用监控工具（如Prometheus），返回结果：

CPU突增时段（02:00-02:10），进程「data-sync-service」占用率达95%（正常时段通常<10%）  
Replanner 再次评估
已定位异常进程，需进一步查该进程日志确认原因，计划无需调整，继续执行步骤2。

执行查进程日志
Executor 调用日志工具，参数更新为进程名=data-sync-service，时间范围=近1小时，返回结果：

日志显示02:00触发全量数据同步任务，遍历数据库1000万条记录，未做分页处理
Replanner 最终评估
已明确根因：全量同步任务未分页导致CPU过载，无需继续执行步骤3（因问题已定位，且历史工单中类似场景解决方案明确），终止任务并返回结论。

最终输出结果
故障根因：服务器进程data-sync-service在02:00执行全量数据同步时，未做分页处理，遍历1000万条记录导致CPU使用率突增。  
建议方案：优化同步逻辑，添加分页参数（如每次拉取1000条），并设置非高峰时段执行。  
核心组件：三个智能角色的协作
就像装修需要 设计师+施工队+监理 ，Plan-Execute-Replan 也靠三个核心智能体（Agent）协同：

Planner（规划器）：任务拆解

作用 ：把用户目标 拆成可执行的步骤清单 ，确保每个步骤清晰、有序。

关键能力 ：理解复杂目标的内在逻辑，生成结构化计划（类似项目甘特图）。

Executor（执行器）：步骤行动

作用 ： 严格执行计划中的当前第一步 ，调用工具（数据库、计算器、API 等）完成具体任务，返回执行结果。

关键能力 ：准确调用工具、处理单步任务（不负责整体规划，只专注做好眼前事）。

Replanner（重规划器）：进度监理

作用 ： 评估 Executor 的执行结果，判断是否需要调整计划

若步骤完成且结果有效：推进到下一个步骤。

若结果缺失/错误（如数据不全、工具调用失败）：修改计划（补充步骤、调整顺序）。

若所有步骤完成：终止任务，返回最终结果。

关键能力 ：判断任务进度、识别执行问题、动态优化计划。`,
		},
	}

	var allDocs []*types.Chunk
	changedCount := 0
	skippedCount := 0

	for _, doc := range docs {
		hash := sha256.Sum256([]byte(doc.content))
		hashStr := hex.EncodeToString(hash[:])
		needsIndex := true
		if qs, ok := store.(*rag.QdrantStore); ok {
			if checkDocHash(qs, doc.uuid, hashStr) {
				needsIndex = false
				skippedCount++
			}
		}
		if needsIndex {
			if qs, ok := store.(*rag.QdrantStore); ok {
				qs.DeleteByDocUUID(context.Background(), doc.uuid)
			}
			chunks := chunker.Chunk(doc.content, doc.title)
			for i := range chunks {
				if chunks[i].Metadata == nil {
					chunks[i].Metadata = make(map[string]string)
				}
				chunks[i].Metadata["doc_uuid"] = doc.uuid
			}
			allDocs = append(allDocs, chunks...)
			changedCount++
		}
	}

	if changedCount == 0 && skippedCount > 0 {
		fmt.Println(" ✅ (all unchanged, skipped)")
		return
	}

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

	if qs, ok := store.(*rag.QdrantStore); ok {
		for _, doc := range docs {
			var docChunks []*types.Chunk
			for _, ch := range allDocs {
				if ch.Metadata != nil && ch.Metadata["doc_uuid"] == doc.uuid {
					docChunks = append(docChunks, ch)
				}
			}
			if len(docChunks) > 0 {
				hash := sha256.Sum256([]byte(doc.content))
				hashStr := hex.EncodeToString(hash[:])
				qs.AddBatchWithMeta(docChunks, doc.uuid, hashStr, true)
			}
		}
	} else {
		store.AddBatch(allDocs)
	}

	gs.BuildFromChunks(allDocs, llmClient)
	fmt.Printf("✅ (%d chunks, %d entities, %d relations)\n",
		len(allDocs),
		func() int { e, _ := gs.Stats(); return e }(),
		func() int { _, r := gs.Stats(); return r }())
}

func showStats(store rag.Store, gs *rag.GraphStore) {
	entities, relations := gs.Stats()
	fmt.Println(strings.Repeat("-", 40))
	fmt.Println("📊 System Status:")
	fmt.Printf("  Chunks in vector store: %d\n", store.Len())
	fmt.Printf("  Graph entities: %d\n", entities)
	fmt.Printf("  Graph relations: %d\n", relations)
	fmt.Println(strings.Repeat("-", 40))
}
