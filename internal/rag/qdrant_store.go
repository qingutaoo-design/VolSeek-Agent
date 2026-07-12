package rag

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

// QdrantStore 基于 Qdrant 向量数据库的持久化存储实现。
type QdrantStore struct {
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
		log.Printf("[Qdrant] Collection %q exists, clearing...", qs.collection)
		// 清空所有点
		_, err := qs.client.Delete(ctx, &qdrant.DeletePoints{
			CollectionName: qs.collection,
			Wait:           qdrant.PtrOf(true),
			Points:         qdrant.NewPointsSelectorFilter(&qdrant.Filter{}),
		})
		if err != nil {
			log.Printf("[Qdrant] Warning clear: %v", err)
		}
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
func (qs *QdrantStore) toPoint(chunk *types.Chunk) *qdrant.PointStruct {
	if chunk.Embedding == nil {
		return nil
	}
	return &qdrant.PointStruct{
		Id: qdrant.NewID(uuid.New().String()),
		Payload: map[string]*qdrant.Value{
			"content":   qdrant.NewValueString(chunk.Content),
			"doc_title": qdrant.NewValueString(chunk.DocTitle),
			"index":     qdrant.NewValueInt(int64(chunk.Index)),
		},
		Vectors: qdrant.NewVectors(float64to32(chunk.Embedding)...),
	}
}

// Add 添加单个分块。
func (qs *QdrantStore) Add(chunk *types.Chunk) {
	qs.chunks = append(qs.chunks, chunk)
	p := qs.toPoint(chunk)
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
func (qs *QdrantStore) AddBatch(chunks []*types.Chunk) {
	qs.chunks = append(qs.chunks, chunks...)

	points := make([]*qdrant.PointStruct, 0, len(chunks))
	for _, ch := range chunks {
		if p := qs.toPoint(ch); p != nil {
			points = append(points, p)
		}
	}
	if len(points) == 0 {
		return
	}

	// 分批 upsert（每批最多 100）
	batchSize := 100
	for i := 0; i < len(points); i += batchSize {
		end := i + batchSize
		if end > len(points) {
			end = len(points)
		}
		_, err := qs.client.Upsert(context.Background(), &qdrant.UpsertPoints{
			CollectionName: qs.collection,
			Wait:           qdrant.PtrOf(true),
			Points:         points[i:end],
		})
		if err != nil {
			log.Printf("[Qdrant] batch upsert error at %d: %v", i, err)
		}
		time.Sleep(50 * time.Millisecond)
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
	r := make([]*types.Chunk, len(qs.chunks))
	copy(r, qs.chunks)
	return r
}

// Len 返回存储中的分块总数。
func (qs *QdrantStore) Len() int {
	return len(qs.chunks)
}

// Close 关闭 gRPC 连接。
func (qs *QdrantStore) Close() error {
	return qs.client.Close()
}
