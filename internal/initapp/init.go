package initapp

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/agent"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/config"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/llm"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/memory"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/rag"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/tools"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

func InitAgent(ctx context.Context) (*agent.AgentEngine, rag.Store, *rag.GraphStore, *llm.Client) {
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
		} else { store = qs; fmt.Println("✅ (Qdrant)"); fmt.Print("🔄 Loading into Qdrant... ") }
	} else { store = rag.NewVectorStore() }

	graphStore := rag.NewGraphStore()
	queryRouter := rag.NewQueryRouter(llmClient)

	rerankerAPIKey := config.GetEnv("RERANK_API_KEY", "")
	rerankerBaseURL := config.GetEnv("RERANK_BASE_URL", "")
	rerankerModel := config.GetEnv("RERANK_MODEL", "")
	reranker := rag.NewReranker(rerankerAPIKey, rerankerBaseURL, rerankerModel)
	if rerankerAPIKey != "" && rerankerBaseURL != "" && rerankerModel != "" {
		fmt.Println("  🔁 Reranker enabled:", rerankerModel)
	} else { fmt.Println("  🔁 Reranker: disabled") }

	retriever := rag.NewRetriever(store, llmClient, graphStore, reranker, 5)
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

	fmt.Print("🔄 Initializing memory... ")
	mem := memory.NewBuffer(100)
	mem.SetSummarizer(func(ctx context.Context, entries []memory.Entry) (string, error) {
		var lines []string
		for _, e := range entries { lines = append(lines, e.Role+": "+e.Content) }
		content := strings.Join(lines, "\n")
		prompt := "Summarize the following conversation into 2-3 sentences:\n" + content
		resp, err := llmClient.Chat(ctx, []types.LLMMessage{{Role: "user", Content: prompt}}, nil, 0.3)
		if err != nil { return "", err }
		if resp != nil { return resp.Content, nil }
		return "", nil
	})
	fmt.Println("✅")

	fmt.Print("🔄 Initializing Agent engine... ")
	agentCfg := &types.AgentConfig{
		Temperature: 0.7, MaxPlanningRounds: 10,
		EnableReflection: true, EnableGraphRAG: true,
		ParallelToolCalls: true, MaxContextTokens: 8192,
	}
	volseek := agent.NewAgentEngine(agentCfg, llmClient, registry, queryRouter, retriever, mem)
	fmt.Println("✅")

	IndexSampleDocuments(chunker, llmClient, store, graphStore)
	return volseek, store, graphStore, llmClient
}