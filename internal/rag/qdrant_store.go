package rag

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

// QdrantStore 基于 Qdrant 向量数据库的持久化存储实现。
type QdrantStore struct {
	mu         sync.RWMutex
	client     *qdrant.Client
	collection string
	dimension  uint64
	chunks     []*types.Chunk
}

// NewQdrantStore 创建 QdrantStore，自动创建或复用 collection。
func NewQdrantStore(ctx context.Context, host string, collection string, dimension uint64) (*QdrantStore, error) {
	client, err := qdrant.NewClient(&qdrant.Config{
		Host: host,
		Port: 6334,
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant connect: %w", err)
	}

	qs := &QdrantStore{
		client:     client,
		collection: collection,
		dimension:  dimension,
		chunks:     make([]*types.Chunk, 0),
	}

	if err := qs.ensureCollection(ctx); err != nil {
		return nil, fmt.Errorf("qdrant init: %w", err)
	}

	return qs, nil
}

func (qs *QdrantStore) ensureCollection(ctx context.Context) error {
	exists, err := qs.client.CollectionExists(ctx, qs.collection)
	if err != nil {
		return err
	}
	if exists {
		log.Printf("[Qdrant] Collection %q exists, reusing existing data", qs.collection)
		return nil
	}

	// 创建 collection
	err = qs.client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: qs.collection,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     qs.dimension,
			Distance: qdrant.Distance_Cosine,
			OnDisk:   qdrant.PtrOf(false),
		}),
		HnswConfig: &qdrant.HnswConfigDiff{
			M:                   qdrant.PtrOf(uint64(16)),
			EfConstruct:         qdrant.PtrOf(uint64(100)),
			FullScanThreshold:   qdrant.PtrOf(uint64(10000)),
		},
	})
	if err != nil {
		return fmt.Errorf("create collection: %w", err)
	}
	log.Printf("[Qdrant] Collection %q created (dim=%d)", qs.collection, qs.dimension)
	return nil
}

// float64to32 转换 float64 切片为 float32。
func float64to32(src []float64) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = float32(v)
	}
	return dst
}

// pointToChunk 将 Qdrant 查询结果转为 Chunk。
func pointToChunk(point *qdrant.ScoredPoint) *types.Chunk {
	if point == nil {
		return nil
	}
	content := ""
	if v, ok := point.Payload["content"]; ok {
		content = v.GetStringValue()
	}
	docTitle := ""
	if v, ok := point.Payload["doc_title"]; ok {
		docTitle = v.GetStringValue()
	}
	index := 0
	if v, ok := point.Payload["index"]; ok {
		index = int(v.GetIntegerValue())
	}
	id := ""
	if point.Id != nil {
		id = point.Id.GetUuid()
	}
	return &types.Chunk{
		ID:       id,
		Content:  content,
		Index:    index,
		DocTitle: docTitle,
	}
}

// toPoint 将 Chunk 转为 Qdrant PointStruct。
func (qs *QdrantStore) toPoint(chunk *types.Chunk, docUUID, docHash string) *qdrant.PointStruct {
	if chunk.Embedding == nil {
		return nil
	}
	payload := map[string]*qdrant.Value{
		"content":   qdrant.NewValueString(chunk.Content),
		"doc_title": qdrant.NewValueString(chunk.DocTitle),
		"doc_uuid":  qdrant.NewValueString(docUUID),
		"doc_hash":  qdrant.NewValueString(docHash),
		"index":     qdrant.NewValueInt(int64(chunk.Index)),
	}
	return &qdrant.PointStruct{
		Id:      qdrant.NewID(uuid.New().String()),
		Payload: payload,
		Vectors: qdrant.NewVectors(float64to32(chunk.Embedding)...),
	}
}

// Add 添加单个分块。
func (qs *QdrantStore) Add(chunk *types.Chunk) {
	qs.mu.Lock()
	qs.chunks = append(qs.chunks, chunk)
	qs.mu.Unlock()
	p := qs.toPoint(chunk, "", "")
	if p == nil {
		return
	}
	_, err := qs.client.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: qs.collection,
		Wait:           qdrant.PtrOf(true),
		Points:         []*qdrant.PointStruct{p},
	})
	if err != nil {
		log.Printf("[Qdrant] upsert error: %v", err)
	}
}

// AddBatch 批量添加分块。
// docUUID 和 docHash 为文档级元数据，用于后续增量更新。
func (qs *QdrantStore) AddBatch(chunks []*types.Chunk) {
	qs.AddBatchWithMeta(chunks, "", "", true)
}

// AddBatchWithMeta 批量添加并写入文档元数据。
// wait=true 同步刷盘；wait=false 异步写入（更快，但宕机可能丢数据）。
func (qs *QdrantStore) AddBatchWithMeta(chunks []*types.Chunk, docUUID, docHash string, wait bool) {
	qs.mu.Lock()
	qs.chunks = append(qs.chunks, chunks...)
	qs.mu.Unlock()

	points := make([]*qdrant.PointStruct, 0, len(chunks))
	for _, ch := range chunks {
		if p := qs.toPoint(ch, docUUID, docHash); p != nil {
			points = append(points, p)
		}
	}
	if len(points) == 0 {
		return
	}

	// 批量写入，每批 5000（Qdrant 建议 ≤ 10000）
	batchSize := 5000
	for i := 0; i < len(points); i += batchSize {
		end := i + batchSize
		if end > len(points) {
			end = len(points)
		}
		_, err := qs.client.Upsert(context.Background(), &qdrant.UpsertPoints{
			CollectionName: qs.collection,
			Wait:           qdrant.PtrOf(wait),
			Points:         points[i:end],
		})
		if err != nil {
			log.Printf("[Qdrant] batch upsert error at %d: %v", i, err)
		}
	}
}

// Search 执行向量搜索。
func (qs *QdrantStore) Search(queryEmbed []float64, topK int, threshold float64) []*types.SearchResult {
	results, err := qs.client.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: qs.collection,
		Query:          qdrant.NewQueryDense(float64to32(queryEmbed)),
		Limit:          qdrant.PtrOf(uint64(topK)),
		ScoreThreshold: qdrant.PtrOf(float32(threshold)),
		WithPayload:    qdrant.NewWithPayload(true),
		Params: &qdrant.SearchParams{
			HnswEf: qdrant.PtrOf(uint64(128)),
			Exact:  qdrant.PtrOf(false),
		},
	})
	if err != nil {
		log.Printf("[Qdrant] search error: %v", err)
		return nil
	}

	out := make([]*types.SearchResult, 0, len(results))
	for _, r := range results {
		chunk := pointToChunk(r)
		if chunk != nil {
			out = append(out, &types.SearchResult{
				Chunk:  chunk,
				Score:  float64(r.Score),
				Method: "vector",
			})
		}
	}
	return out
}

// GetAllChunks 返回所有分块的副本。
func (qs *QdrantStore) GetAllChunks() []*types.Chunk {
	qs.mu.RLock()
	defer qs.mu.RUnlock()
	r := make([]*types.Chunk, len(qs.chunks))
	copy(r, qs.chunks)
	return r
}

// Len 返回存储中的分块总数。
func (qs *QdrantStore) Len() int {
	qs.mu.RLock()
	defer qs.mu.RUnlock()
	return len(qs.chunks)
}

// ScrollByFilter 按 payload 字段过滤查询点，用于验证文档哈希。
func (qs *QdrantStore) ScrollByFilter(ctx context.Context, filter map[string]string, limit uint32) ([]*qdrant.RetrievedPoint, error) {
	var conditions []*qdrant.Condition
	for k, v := range filter {
		conditions = append(conditions, qdrant.NewMatch(k, v))
	}
	resp, err := qs.client.Scroll(context.Background(), &qdrant.ScrollPoints{
		CollectionName: qs.collection,
		Filter: &qdrant.Filter{
			Must: conditions,
		},
		Limit:       &limit,
		WithPayload: qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// Count 返回 Qdrant 集合中的实际点数。
func (qs *QdrantStore) Count(ctx context.Context) (uint64, error) {
	return qs.client.Count(ctx, &qdrant.CountPoints{
		CollectionName: qs.collection,
		Exact:          qdrant.PtrOf(true),
	})
}

// DeleteByDocUUID 删除指定文档的所有关联 chunk（用于增量更新）。
func (qs *QdrantStore) DeleteByDocUUID(ctx context.Context, docUUID string) error {
	_, err := qs.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: qs.collection,
		Wait:           qdrant.PtrOf(true),
		Points: qdrant.NewPointsSelectorFilter(&qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewMatch("doc_uuid", docUUID),
			},
		}),
	})
	// 同步清除本地缓存
	qs.mu.Lock()
	var remaining []*types.Chunk
	for _, ch := range qs.chunks {
		if ch.Metadata == nil || ch.Metadata["doc_uuid"] != docUUID {
			remaining = append(remaining, ch)
		}
	}
	qs.chunks = remaining
	qs.mu.Unlock()
	return err
}
