package initapp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/llm"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/rag"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

type docEntry struct{ uuid, title, content string }

func IndexSampleDocuments(chunker *rag.Chunker, llmClient *llm.Client, store rag.Store, gs *rag.GraphStore) {
	fmt.Print("🔄 Indexing sample documents... ")
	docs := sampleDocs()
	var allDocs []*types.Chunk
	changed, skipped := 0, 0
	for _, doc := range docs {
		hash := sha256.Sum256([]byte(doc.content))
		hashStr := hex.EncodeToString(hash[:])
		need := true
		if qs, ok := store.(*rag.QdrantStore); ok {
			if checkDocHash(qs, doc.uuid, hashStr) { need = false; skipped++ }
		}
		if need {
			if qs, ok := store.(*rag.QdrantStore); ok { qs.DeleteByDocUUID(context.Background(), doc.uuid) }
			chunks := chunker.Chunk(doc.content, doc.title)
			for i := range chunks { if chunks[i].Metadata == nil { chunks[i].Metadata = make(map[string]string) }; chunks[i].Metadata["doc_uuid"] = doc.uuid }
			allDocs = append(allDocs, chunks...); changed++
		}
	}
	if changed == 0 && skipped > 0 { fmt.Println(" ✅ (all unchanged, skipped)"); return }
	texts := make([]string, len(allDocs))
	for i, ch := range allDocs { texts[i] = ch.Content }
	if len(texts) > 0 {
		emb, err := llmClient.EmbedBatch(context.Background(), texts)
		if err != nil { log.Printf("Warning: embedding failed: %v", err) } else { for i, e := range emb { if e != nil { allDocs[i].Embedding = e } } }
	}
	if qs, ok := store.(*rag.QdrantStore); ok {
		for _, doc := range docs {
			var dc []*types.Chunk
			for _, ch := range allDocs { if ch.Metadata != nil && ch.Metadata["doc_uuid"] == doc.uuid { dc = append(dc, ch) } }
			if len(dc) > 0 { h := sha256.Sum256([]byte(doc.content)); qs.AddBatchWithMeta(dc, doc.uuid, hex.EncodeToString(h[:]), true) }
		}
	} else { store.AddBatch(allDocs) }
	gs.BuildFromChunks(allDocs, llmClient)
	fmt.Printf("✅ (%d chunks, %d entities, %d relations)\n", len(allDocs),
		func() int { e, _ := gs.Stats(); return e }(), func() int { _, r := gs.Stats(); return r }())
}

func checkDocHash(qs *rag.QdrantStore, docUUID, expectedHash string) bool {
	points, err := qs.ScrollByFilter(context.Background(), map[string]string{"doc_uuid": docUUID}, 1)
	if err != nil || len(points) == 0 { return false }
	if v, ok := points[0].Payload["doc_hash"]; ok { return v.GetStringValue() == expectedHash }
	return false
}

func sampleDocs() []docEntry {
	return []docEntry{
		{"doc-rag-intro", "RAG 技术介绍",
			`RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的AI技术。RAG 的核心思想是在 LLM 生成答案前先从外部知识库中检索相关信息。GraphRAG 是 RAG 的进阶版本，通过构建知识图谱来理解实体间的关系。`},
		{"doc-go-intro", "Go 语言入门",
			`Go（又称 Golang）是 Google 开发的开源编程语言。Go 的三大设计目标是简洁的语法、高效的编译、强大的并发支持。goroutine 是轻量级线程，channel 是 goroutine 之间的通信机制。`},
		{"doc-volseek", "VolSeek-Agent 设计文档",
			`VolSeek-Agent 是一个具有规划、执行和反思能力的 RAG Agent 框架。核心架构：Plan → Execute → Reflect。支持 tools：knowledge_search, graph_search, web_search, calculator。`},
	}
}
