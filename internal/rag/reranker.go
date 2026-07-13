// Package rag 提供完整的 RAG（检索增强生成）能力。
package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"sort"
	"time"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

// Reranker 对检索结果进行二次重排，提升排序质量。
// 向量搜索用双编码器（bi-encoder）快速召回，重排器用交叉编码器（cross-encoder）精排。
type Reranker interface {
	// Rerank 对候选项重排，按相关性降序返回。
	// query: 用户原始查询
	// candidates: 初筛结果（来自向量/关键词检索）
	// returns: 按 reranker score 降序排列的结果
	Rerank(ctx context.Context, query string, candidates []*types.SearchResult) []*types.SearchResult
}

// noopReranker 不执行重排，原样返回（兜底）。
type noopReranker struct{}

func (r *noopReranker) Rerank(_ context.Context, _ string, candidates []*types.SearchResult) []*types.SearchResult {
	return candidates
}

// APIReranker 调用外部 API 做交叉编码重排。
// 兼容 SiliconFlow 的 rerank 接口，也兼容 Cohere / Jina 等。
type APIReranker struct {
	apiKey    string
	baseURL   string
	model     string
	client    *http.Client
}

// rerankRequest 和 rerankResponse 对应 SiliconFlow rerank 接口的请求/响应格式。
type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
}

type rerankResponse struct {
	Results []rerankResult `json:"results"`
}

type rerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

// NewReranker 根据配置创建重排器。
// apiKey / baseURL / model 任意一个为空时返回 noopReranker（不执行重排）。
func NewReranker(apiKey, baseURL, model string) Reranker {
	if apiKey == "" || baseURL == "" || model == "" {
		log.Println("[Reranker] not configured, skipping rerank")
		return &noopReranker{}
	}
	return &APIReranker{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		client: &http.Client{
			Timeout: 30 * time.Second, // 重排超时 30 秒
		},
	}
}

// Rerank 调用外部 API 对候选文档重排。
func (r *APIReranker) Rerank(ctx context.Context, query string, candidates []*types.SearchResult) []*types.SearchResult {
	if len(candidates) == 0 {
		return candidates
	}

	// 提取文档内容列表
	docs := make([]string, len(candidates))
	for i, c := range candidates {
		if c.Chunk != nil {
			docs[i] = c.Chunk.Content
		}
	}

	// 构造请求
	reqBody := rerankRequest{
		Model:     r.model,
		Query:     query,
		Documents: docs,
	}
	bodyBytes, _ := json.Marshal(reqBody)

	req, err := http.NewRequestWithContext(ctx, "POST", r.baseURL, bytes.NewReader(bodyBytes))
	if err != nil {
		log.Printf("[Reranker] create request failed: %v", err)
		return candidates
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+r.apiKey)

	// 发送请求
	resp, err := r.client.Do(req)
	if err != nil {
		log.Printf("[Reranker] API call failed: %v", err)
		return candidates
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[Reranker] read response failed: %v", err)
		return candidates
	}

	var rerankResp rerankResponse
	if err := json.Unmarshal(respBody, &rerankResp); err != nil {
		log.Printf("[Reranker] parse response failed: %v", err)
		return candidates
	}

	if len(rerankResp.Results) == 0 {
		log.Println("[Reranker] empty results, returning original order")
		return candidates
	}

	// 按重排分数降序排列，同时更新 Score 和 Method
	scored := make([]scoredResult, len(rerankResp.Results))
	for i, r := range rerankResp.Results {
		if r.Index >= 0 && r.Index < len(candidates) {
			scored[i] = scoredResult{
				result: candidates[r.Index],
				score:  r.RelevanceScore,
			}
		}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	out := make([]*types.SearchResult, len(scored))
	for i, s := range scored {
		s.result.Score = s.score
		s.result.Method = "rerank"
		out[i] = s.result
	}

	log.Printf("[Reranker] reranked %d items, top score: %.4f", len(out), out[0].Score)
	return out
}

// scoredResult 内部辅助类型，用于分数排序。
type scoredResult struct {
	result *types.SearchResult
	score  float64
}
