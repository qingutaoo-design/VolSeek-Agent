// Package rag 提供完整的 RAG（检索增强生成）能力。
// 包含三大核心组件：
//  1. Chunker：智能文档分块
//  2. Retriever：多策略检索器（向量搜索 + 知识图谱 + 关键词）
//  3. QueryRouter：查询意图路由，自动选择最优检索策略
package rag

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/llm"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

// ============================================================================
// Chunker — 文档分块器
// ============================================================================

// Chunker 支持两种分块策略：固定大小重叠分块 和 语义段落分块。
// 策略选择：对结构化文档（Markdown、代码）使用段落分块保留语义边界；
// 对纯文本使用固定大小分块确保检索粒度。
type Chunker struct {
	config types.ChunkConfig
}

// NewChunker 创建分块器，config 为 nil 时使用默认值（每块 500 字符，重叠 50）。
func NewChunker(config *types.ChunkConfig) *Chunker {
	if config == nil {
		return &Chunker{
			config: types.ChunkConfig{Size: 500, Overlap: 50},
		}
	}
	return &Chunker{config: *config}
}

// Chunk 自动选择分块策略。
// 对 Markdown 文本使用段落分块，其余使用固定大小分块。
func (c *Chunker) Chunk(text, title string) []*types.Chunk {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	// 检测是否为 Markdown（包含标题标记）
	if strings.Contains(text, "# ") || strings.Contains(text, "## ") {
		return c.chunkByMarkdown(text, title)
	}
	return c.chunkByFixedSize(text, title)
}

// chunkByFixedSize 固定大小重叠分块。
// 这是最通用的分块策略，保证每块不超过 Size 个字符。
func (c *Chunker) chunkByFixedSize(text, title string) []*types.Chunk {
	runes := []rune(text)
	totalLen := len(runes)

	if totalLen == 0 {
		return nil
	}
	if totalLen <= c.config.Size {
		return []*types.Chunk{{
			ID:       fmt.Sprintf("%s-0", title),
			Content:  text,
			Index:    0,
			DocTitle: title,
		}}
	}

	var chunks []*types.Chunk
	start := 0
	index := 0

	for start < totalLen {
		end := start + c.config.Size
		if end > totalLen {
			end = totalLen
		}

		chunkRunes := runes[start:end]

		// 如果不在文本末尾，尝试在句子边界截断
		if end < totalLen {
			// 从后往前找句号、换行等自然边界
			for j := len(chunkRunes) - 1; j > start+c.config.Size/2; j-- {
				if chunkRunes[j] == '.' || chunkRunes[j] == '。' ||
					chunkRunes[j] == '\n' || chunkRunes[j] == '!' || chunkRunes[j] == '？' {
					end = start + j + 1
					chunkRunes = runes[start:end]
					break
				}
			}
		}

		chunks = append(chunks, &types.Chunk{
			ID:       fmt.Sprintf("%s-%d", title, index),
			Content:  string(chunkRunes),
			Index:    index,
			DocTitle: title,
		})
		index++

		// 移动窗口（带重叠）
		nextStart := end - c.config.Overlap
		if nextStart <= start {
			nextStart = end
		}
		start = nextStart
	}

	return chunks
}

// chunkByMarkdown 按 Markdown 段落分块，保留标题层级信息。
// 比固定分块更适合技术文档，因为语义完整的段落不会被截断。
func (c *Chunker) chunkByMarkdown(text, title string) []*types.Chunk {
	lines := strings.Split(text, "\n")
	var chunks []*types.Chunk
	var currentBuilder strings.Builder
	var currentHeading string
	currentSize := 0
	index := 0

	flush := func() {
		if currentBuilder.Len() > 0 {
			content := strings.TrimSpace(currentBuilder.String())
			if content != "" {
				chunkTitle := title
				if currentHeading != "" {
					chunkTitle = fmt.Sprintf("%s > %s", title, currentHeading)
				}
				chunks = append(chunks, &types.Chunk{
					ID:       fmt.Sprintf("%s-%d", title, index),
					Content:  content,
					Index:    index,
					DocTitle: chunkTitle,
				})
				index++
			}
			currentBuilder.Reset()
			currentSize = 0
		}
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Markdown 标题是新段落的开始
		if strings.HasPrefix(trimmed, "# ") || strings.HasPrefix(trimmed, "## ") ||
			strings.HasPrefix(trimmed, "### ") {
			flush()
			currentHeading = strings.TrimLeft(trimmed, "# ")
		}

		lineLen := utf8.RuneCountInString(line)
		if currentSize+lineLen > c.config.Size && currentSize > 0 {
			flush()
		}

		currentBuilder.WriteString(line)
		currentBuilder.WriteString("\n")
		currentSize += lineLen + 1
	}

	// 刷新最后一个段落
	flush()

	if len(chunks) == 0 {
		// 如果没有成功分块（如短文本），按固定大小分块兜底
		return c.chunkByFixedSize(text, title)
	}

	return chunks
}

// ============================================================================
// VectorStore — 内存向量存储
// ============================================================================

// VectorStore 是一个线程安全的内存向量存储。
// 提供向量的增、删、查能力。适合中小规模场景（<10万条）。
// 生产环境建议替换为 pgvector / Milvus 等专业向量数据库。
type VectorStore struct {
	mu     sync.RWMutex
	chunks []*types.Chunk
	// docIndex 加速按文档查找
	docIndex map[string][]*types.Chunk
}

// NewVectorStore 创建内存向量存储。
func NewVectorStore() *VectorStore {
	return &VectorStore{
		chunks:   make([]*types.Chunk, 0),
		docIndex: make(map[string][]*types.Chunk),
	}
}

// Add 添加一个分块到向量存储。
func (vs *VectorStore) Add(chunk *types.Chunk) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.chunks = append(vs.chunks, chunk)
	vs.docIndex[chunk.DocTitle] = append(vs.docIndex[chunk.DocTitle], chunk)
}

// AddBatch 批量添加分块。
func (vs *VectorStore) AddBatch(chunks []*types.Chunk) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	for _, chunk := range chunks {
		vs.chunks = append(vs.chunks, chunk)
		vs.docIndex[chunk.DocTitle] = append(vs.docIndex[chunk.DocTitle], chunk)
	}
}

// Search 在内存中执行暴力向量搜索。
// 计算所有向量与查询向量的余弦相似度，返回 Top-K。
// 注意：O(n) 时间复杂度，n 较大时建议用 HNSW/IVF 索引。
func (vs *VectorStore) Search(queryEmbed []float64, topK int, threshold float64) []*types.SearchResult {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	var results []*types.SearchResult

	for _, chunk := range vs.chunks {
		if chunk.Embedding == nil {
			continue
		}
		score := cosineSimilarity(queryEmbed, chunk.Embedding)
		if score >= threshold {
			results = append(results, &types.SearchResult{
				Chunk:  chunk,
				Score:  score,
				Method: "vector",
			})
		}
	}

	// 按相似度降序排列
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > topK {
		results = results[:topK]
	}

	return results
}

// Len 返回存储中的分块总数。
func (vs *VectorStore) Len() int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	return len(vs.chunks)
}

// ============================================================================
// Retriever — 多策略检索器
// ============================================================================

// Retriever 整合了向量搜索、关键词搜索和知识图谱查询。
// 根据 QueryRouter 的判断，选择最优检索策略或混合使用。
type Retriever struct {
	store    *VectorStore
	llm      *llm.Client
	graph    *GraphStore
	topK     int
	theshold float64
}

// NewRetriever 创建检索器。
func NewRetriever(store *VectorStore, llmClient *llm.Client, graph *GraphStore, topK int) *Retriever {
	return &Retriever{
		store:    store,
		llm:      llmClient,
		graph:    graph,
		topK:     topK,
		theshold: 0.5,
	}
}

// Retrieve 根据查询意图执行多策略检索。
// 策略选择：
//   - factual: 关键词搜索优先（精确匹配实体名）
//   - conceptual: 向量搜索优先（语义相似度）
//   - comparative: 混合搜索（向量+关键词）
//   - analytical: 向量搜索 + 知识图谱
//   - recent: 不做检索（需要联网）
//   - unknown: 混合搜索兜底
func (r *Retriever) Retrieve(ctx context.Context, query string, intent *types.QueryIntent) ([]*types.SearchResult, error) {
	switch intent.Type {
	case types.QueryFactual:
		// 事实性查询：关键词搜索 + 向量搜索并行
		results := r.hybridSearch(ctx, query, r.topK)
		return r.deduplicate(results), nil

	case types.QueryConceptual:
		// 概念性查询：纯向量搜索
		embed, err := r.llm.Embed(ctx, query)
		if err != nil {
			return nil, err
		}
		return r.store.Search(embed, r.topK, r.theshold), nil

	case types.QueryComparative:
		// 比较性查询：混合搜索（向量+关键词 RRF 融合）
		results := r.hybridSearch(ctx, query, r.topK*2)
		results = applyRRF(results, 60)
		if len(results) > r.topK {
			results = results[:r.topK]
		}
		return r.deduplicate(results), nil

	case types.QueryAnalytical:
		// 分析性查询：向量搜索 + 知识图谱拓展
		embed, err := r.llm.Embed(ctx, query)
		if err != nil {
			return nil, err
		}
		vecResults := r.store.Search(embed, r.topK, r.theshold)

		// 用知识图谱拓展结果
		if r.graph != nil && len(vecResults) > 0 {
			graphResults := r.expandWithGraph(vecResults)
			vecResults = append(vecResults, graphResults...)
		}

		return r.deduplicate(vecResults), nil

	case types.QueryRecent:
		// 实时性查询：返回空结果（需要 web_search 工具）
		return nil, nil

	default:
		// 兜底：混合搜索
		results := r.hybridSearch(ctx, query, r.topK)
		return r.deduplicate(results), nil
	}
}

// hybridSearch 同时执行向量搜索和关键词搜索。
func (r *Retriever) hybridSearch(ctx context.Context, query string, topK int) []*types.SearchResult {
	// 1. 向量搜索
	embed, err := r.llm.Embed(ctx, query)
	var vecResults []*types.SearchResult
	if err == nil {
		vecResults = r.store.Search(embed, topK, r.theshold)
	}

	// 2. 关键词搜索（BM25 风格：简单词频匹配）
	kwResults := r.keywordSearch(query, topK)

	// 3. 融合结果
	all := append(vecResults, kwResults...)
	return all
}

// keywordSearch 简单的关键词搜索（词袋模型 + TF 排序）。
func (r *Retriever) keywordSearch(query string, topK int) []*types.SearchResult {
	queryTokens := tokenize(query)

	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	type scored struct {
		result *types.SearchResult
		score  float64
	}

	var results []scored

	for _, chunk := range r.store.chunks {
		chunkTokens := tokenize(chunk.Content)
		matchCount := 0
		for _, qt := range queryTokens {
			for _, ct := range chunkTokens {
				if strings.EqualFold(qt, ct) {
					matchCount++
					break
				}
			}
		}
		if matchCount > 0 {
			score := float64(matchCount) / float64(len(queryTokens))
			results = append(results, scored{
				result: &types.SearchResult{
					Chunk:  chunk,
					Score:  score,
					Method: "keyword",
				},
				score: score,
			})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if len(results) > topK {
		results = results[:topK]
	}

	out := make([]*types.SearchResult, len(results))
	for i, r := range results {
		out[i] = r.result
	}
	return out
}

// expandWithGraph 用知识图谱中的关联实体拓展检索结果。
// 当找到某个实体的相关信息后，沿着关系边找到关联实体，返回它们的上下文。
func (r *Retriever) expandWithGraph(results []*types.SearchResult) []*types.SearchResult {
	if r.graph == nil {
		return nil
	}

	var extra []*types.SearchResult
	seen := make(map[string]bool)

	for _, res := range results {
		if res.Chunk == nil {
			continue
		}

		// 从结果中抽取实体名（简化版：用 chunk 标题中的词汇）
		entities := r.graph.FindEntities(res.Chunk.DocTitle)
		for _, entity := range entities {
			if seen[entity.ID] {
				continue
			}
			seen[entity.ID] = true

			// 沿着关系找相邻实体
			neighbors := r.graph.GetNeighbors(entity.ID, 1)
			for _, neighbor := range neighbors {
				// 找邻居实体的源文档块
				for _, chunkID := range neighbor.ChunkIDs {
					_ = chunkID
					extra = append(extra, &types.SearchResult{
						Chunk: &types.Chunk{
							ID:      neighbor.ID,
							Content: neighbor.Context,
							DocTitle: neighbor.Name,
						},
						Score:  0.6, // 知识图谱关系的固定置信度
						Method: "graph",
					})
				}
			}
		}
	}

	return extra
}

// deduplicate 按 chunk ID 去重，保留得分最高的结果。
func (r *Retriever) deduplicate(results []*types.SearchResult) []*types.SearchResult {
	seen := make(map[string]*types.SearchResult)
	for _, res := range results {
		if res.Chunk == nil {
			continue
		}
		id := res.Chunk.ID
		if existing, ok := seen[id]; ok {
			if res.Score > existing.Score {
				seen[id] = res
			}
		} else {
			seen[id] = res
		}
	}

	out := make([]*types.SearchResult, 0, len(seen))
	for _, res := range seen {
		out = append(out, res)
	}

	sort.Slice(out, func(i, j int) bool {
		return out[i].Score > out[j].Score
	})

	return out
}

// ============================================================================
// GraphStore — 知识图谱存储
// ============================================================================

// GraphStore 管理实体-关系知识图谱。
// 图谱通过分析文档内容自动构建，支持实体消歧和关系遍历。
type GraphStore struct {
	mu        sync.RWMutex
	entities  map[string]*types.EntityNode // id -> entity
	relations []*types.Relation
	// nameIndex 加速按名查找
	nameIndex map[string]*types.EntityNode
}

// NewGraphStore 创建知识图谱存储。
func NewGraphStore() *GraphStore {
	return &GraphStore{
		entities:  make(map[string]*types.EntityNode),
		nameIndex: make(map[string]*types.EntityNode),
	}
}

// AddEntity 添加实体节点。
func (gs *GraphStore) AddEntity(entity *types.EntityNode) {
	gs.mu.Lock()
	defer gs.mu.Unlock()
	gs.entities[entity.ID] = entity
	gs.nameIndex[entity.Name] = entity
}

// AddRelation 添加关系边。
func (gs *GraphStore) AddRelation(rel *types.Relation) {
	gs.mu.Lock()
	defer gs.mu.Unlock()
	gs.relations = append(gs.relations, rel)
}

// FindEntities 按名称查找实体。
func (gs *GraphStore) FindEntities(name string) []*types.EntityNode {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	var results []*types.EntityNode
	lower := strings.ToLower(name)

	for _, entity := range gs.entities {
		if strings.Contains(strings.ToLower(entity.Name), lower) ||
			strings.Contains(lower, strings.ToLower(entity.Name)) {
			results = append(results, entity)
		}
	}
	return results
}

// GetNeighbors 获取指定实体的 N 跳邻居。
func (gs *GraphStore) GetNeighbors(entityID string, hops int) []*types.EntityNode {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	visited := make(map[string]bool)
	var result []*types.EntityNode

	var dfs func(id string, depth int)
	dfs = func(id string, depth int) {
		if depth > hops || visited[id] {
			return
		}
		visited[id] = true

		for _, rel := range gs.relations {
			var neighborID string
			if rel.SourceID == id {
				neighborID = rel.TargetID
			} else if rel.TargetID == id {
				neighborID = rel.SourceID
			} else {
				continue
			}

			if neighbor, ok := gs.entities[neighborID]; ok && !visited[neighborID] {
				result = append(result, neighbor)
				dfs(neighborID, depth+1)
			}
		}
	}

	dfs(entityID, 0)
	return result
}

// BuildFromChunks 从文档分块中自动提取实体和关系，构建知识图谱。
// 这是 GraphRAG 的核心：分析文档内容，提取关键实体及其关系。
func (gs *GraphStore) BuildFromChunks(chunks []*types.Chunk, llmClient *llm.Client) error {
	// 用 LLM 从文档块中抽取实体关系
	// 注意：此处简化处理，实际生产应用需更复杂的 NER 和关系抽取
	for _, chunk := range chunks {
		extracted := extractEntitiesFromContent(chunk.Content, chunk.ID)
		for _, entity := range extracted {
			gs.AddEntity(entity)
		}
	}

	// 在同一文档中出现的实体之间建立共现关系
	gs.buildCooccurrenceRelations(chunks)

	return nil
}

// buildCooccurrenceRelations 在同一文档块中出现的实体之间建立"co-occurs-with"关系。
func (gs *GraphStore) buildCooccurrenceRelations(chunks []*types.Chunk) {
	for _, chunk := range chunks {
		entities := gs.FindEntities(chunk.DocTitle)
		for i := 0; i < len(entities); i++ {
			for j := i + 1; j < len(entities); j++ {
				gs.AddRelation(&types.Relation{
					SourceID: entities[i].ID,
					TargetID: entities[j].ID,
					Relation: "co-occurs-with",
					Weight:   0.5,
				})
			}
		}
	}
}

// Stats 返回知识图谱的统计信息。
func (gs *GraphStore) Stats() (entities int, relations int) {
	gs.mu.RLock()
	defer gs.mu.RUnlock()
	return len(gs.entities), len(gs.relations)
}

// ============================================================================
// QueryRouter — 查询意图路由器
// ============================================================================

// QueryRouter 分析用户查询，识别意图类型，决定最优检索策略。
// 创新点：不把查询盲目发给向量搜索，而是先"理解"查询需要什么。
type QueryRouter struct {
	llm *llm.Client
}

// NewQueryRouter 创建查询路由器。
func NewQueryRouter(llmClient *llm.Client) *QueryRouter {
	return &QueryRouter{llm: llmClient}
}

// Analyze 分析查询意图。
// 使用 LLM 做轻量级分类（比基于规则的方法更准确，比二次检索更快速）。
func (qr *QueryRouter) Analyze(ctx context.Context, query string) (*types.QueryIntent, error) {
	// 用 LLM 做查询分类（一次轻量级调用）
	systemPrompt := `You are a query analyzer. Classify the user's question into one of these types:
- factual: Questions asking for specific facts, definitions, or data
- conceptual: Questions asking for explanations of concepts or ideas
- comparative: Questions comparing two or more things
- procedural: Questions about how to do something (step-by-step)
- analytical: Questions requiring analysis, pros/cons, evaluation
- recent: Questions about current events, real-time data, recent information

Respond with ONLY a JSON object:
{"type": "...", "entities": [...], "needs_web": false, "needs_kg": false, "summary": "..."}`

	messages := []types.LLMMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: query},
	}

	resp, err := qr.llm.Chat(ctx, messages, nil, 0.1) // 低温度确保一致性
	if err != nil {
		// LLM 失败时，用启发式方法兜底
		return qr.heuristicAnalyze(query), nil
	}

	// 尝试解析 JSON 响应
	intent := qr.parseIntent(resp.Content)
	if intent == nil {
		return qr.heuristicAnalyze(query), nil
	}

	return intent, nil
}

// heuristicAnalyze 基于规则的启发式查询分析（兜底方案）。
func (qr *QueryRouter) heuristicAnalyze(query string) *types.QueryIntent {
	lower := strings.ToLower(query)
	intent := &types.QueryIntent{
		Keywords: tokenize(query),
	}

	// 事实性查询标记词
	factualMarkers := []string{"什么是", "who is", "what is", "when did", "where is", "define"}
	for _, m := range factualMarkers {
		if strings.Contains(lower, m) {
			intent.Type = types.QueryFactual
			return intent
		}
	}

	// 比较性查询标记词
	compareMarkers := []string{"区别", "difference", "compare", "vs", "versus", "对比"}
	for _, m := range compareMarkers {
		if strings.Contains(lower, m) {
			intent.Type = types.QueryComparative
			return intent
		}
	}

	// 步骤性查询标记词
	procMarkers := []string{"how to", "how do", "步骤", "如何", "ways to"}
	for _, m := range procMarkers {
		if strings.Contains(lower, m) {
			intent.Type = types.QueryProcedural
			return intent
		}
	}

	// 实时性查询标记词
	recentMarkers := []string{"latest", "today", "current", "news", "price", "recent"}
	for _, m := range recentMarkers {
		if strings.Contains(lower, m) {
			intent.Type = types.QueryRecent
			intent.NeedsWeb = true
			return intent
		}
	}

	// 分析性查询（包含优缺点、评价类关键词）
	analysisMarkers := []string{"分析", "评价", "pros", "cons", "advantage", "disadvantage", "优缺点"}
	for _, m := range analysisMarkers {
		if strings.Contains(lower, m) {
			intent.Type = types.QueryAnalytical
			intent.NeedsKG = true
			return intent
		}
	}

	// 默认视为概念性查询
	intent.Type = types.QueryConceptual
	return intent
}

// parseIntent 从 LLM 的 JSON 响应解析 QueryIntent。
func (qr *QueryRouter) parseIntent(jsonStr string) *types.QueryIntent {
	// 尝试从 JSON 中提取
	if idx := strings.Index(jsonStr, "{"); idx >= 0 {
		if end := strings.LastIndex(jsonStr, "}"); end > idx {
			jsonStr = jsonStr[idx : end+1]
		}
	}

	// 简单 JSON 解析（不依赖外部库）
	intent := &types.QueryIntent{}
	content := jsonStr

	// 提取 type
	if t := extractJSONString(content, "type"); t != "" {
		switch t {
		case "factual":
			intent.Type = types.QueryFactual
		case "conceptual":
			intent.Type = types.QueryConceptual
		case "comparative":
			intent.Type = types.QueryComparative
		case "procedural":
			intent.Type = types.QueryProcedural
		case "analytical":
			intent.Type = types.QueryAnalytical
		case "recent":
			intent.Type = types.QueryRecent
			intent.NeedsWeb = true
		default:
			intent.Type = types.QueryConceptual
		}
	} else {
		return nil
	}

	intent.NeedsWeb = extractJSONBool(content, "needs_web")
	intent.NeedsKG = extractJSONBool(content, "needs_kg")
	intent.Summary = extractJSONString(content, "summary")

	return intent
}

// ============================================================================
// 工具函数
// ============================================================================

// cosineSimilarity 计算两个向量的余弦相似度。
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (float64(math.Sqrt(normA)) * float64(math.Sqrt(normB)))
}

// applyRRF 应用 Reciprocal Rank Fusion 融合多个检索结果。
// k 是 RRF 常数，通常取 60。RRF 将多个排序列表合并为一个。
func applyRRF(results []*types.SearchResult, k int) []*types.SearchResult {
	// 按方法分组，在同一方法内计算 rank
	type entry struct {
		result *types.SearchResult
		rrf    float64
	}

	rankMap := make(map[string]map[string]int) // method -> chunkID -> rank
	unique := make(map[string]*types.SearchResult)

	for i, res := range results {
		if res.Chunk == nil {
			continue
		}
		id := res.Chunk.ID
		if _, ok := rankMap[res.Method]; !ok {
			rankMap[res.Method] = make(map[string]int)
		}
		if _, ok := rankMap[res.Method][id]; !ok {
			rankMap[res.Method][id] = i + 1
		}
		unique[id] = res
	}

	// 计算每个结果的 RRF 分数
	var entries []entry
	for id, res := range unique {
		var rrf float64
		for _, ranks := range rankMap {
			if rank, ok := ranks[id]; ok {
				rrf += 1.0 / float64(k+rank)
			}
		}
		entries = append(entries, entry{result: res, rrf: rrf})
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].rrf > entries[j].rrf
	})

	out := make([]*types.SearchResult, len(entries))
	for i, e := range entries {
		out[i] = e.result
		out[i].Score = e.rrf
	}
	return out
}

// tokenize 对文本进行简单的中英文分词。
func tokenize(text string) []string {
	var tokens []string
	// 转为小写
	text = strings.ToLower(text)

	// 按空白字符分割
	words := strings.Fields(text)
	for _, word := range words {
		// 移除标点
		word = strings.Trim(word, ".,!?;:'\"()[]{}<>，。！？；：\u201c\u201d\u2018\u2019（）【】《》")
		if word == "" {
			continue
		}
		tokens = append(tokens, word)
	}

	return tokens
}

// extractEntitiesFromContent 从文本中提取实体（简化版）。
// 生产环境应使用 NER 模型或 LLM 进行分析。
func extractEntitiesFromContent(content, chunkID string) []*types.EntityNode {
	var entities []*types.EntityNode
	seen := make(map[string]bool)

	// 简单启发式：提取大写开头的连续单词作为实体名
	words := strings.Fields(content)
	for i := 0; i < len(words); i++ {
		word := strings.TrimSpace(words[i])
		if len(word) < 2 {
			continue
		}

		// 检测专有名词（大写开头）
		runes := []rune(word)
		if runes[0] >= 'A' && runes[0] <= 'Z' {
			// 尝试组合后续大写开头的词（如 "Machine Learning"）
			var entityWords []string
			entityWords = append(entityWords, word)
			for j := i + 1; j < len(words); j++ {
				nextRunes := []rune(strings.TrimSpace(words[j]))
				if len(nextRunes) == 0 {
					break
				}
				if nextRunes[0] >= 'A' && nextRunes[0] <= 'Z' {
					entityWords = append(entityWords, strings.TrimSpace(words[j]))
					i = j
				} else {
					break
				}
			}

			entityName := strings.Join(entityWords, " ")
			if !seen[entityName] && len(entityName) < 50 {
				seen[entityName] = true
				entities = append(entities, &types.EntityNode{
					ID:       fmt.Sprintf("entity-%d", len(entities)),
					Name:     entityName,
					Type:     "concept",
					Context:  getSurroundingText(content, entityName, 100),
					ChunkIDs: []string{chunkID},
				})
			}
		}
	}

	return entities
}

// getSurroundingText 获取某个词在文本中的上下文。
func getSurroundingText(text, word string, contextLen int) string {
	idx := strings.Index(text, word)
	if idx < 0 {
		return text
	}

	start := idx - contextLen
	if start < 0 {
		start = 0
	}
	end := idx + len(word) + contextLen
	if end > len(text) {
		end = len(text)
	}

	result := text[start:end]
	if start > 0 {
		result = "..." + result
	}
	if end < len(text) {
		result = result + "..."
	}
	return result
}

// extractJSONString 从 JSON 字符串中提取指定 key 的 string 值。
func extractJSONString(json, key string) string {
	pattern := fmt.Sprintf(`"%s"`, key)
	idx := strings.Index(json, pattern)
	if idx < 0 {
		return ""
	}
	// 找冒号
	colonIdx := strings.Index(json[idx+len(pattern):], ":")
	if colonIdx < 0 {
		return ""
	}
	valueStart := idx + len(pattern) + colonIdx + 1
	// 找引号
	quoteStart := strings.Index(json[valueStart:], `"`)
	if quoteStart < 0 {
		return ""
	}
	quoteStart += valueStart + 1
	quoteEnd := strings.Index(json[quoteStart:], `"`)
	if quoteEnd < 0 {
		return ""
	}
	return json[quoteStart : quoteStart+quoteEnd]
}

// extractJSONBool 从 JSON 字符串中提取指定 key 的 bool 值。
func extractJSONBool(json, key string) bool {
	pattern := fmt.Sprintf(`"%s"`, key)
	idx := strings.Index(json, pattern)
	if idx < 0 {
		return false
	}
	rest := json[idx+len(pattern):]
	colonIdx := strings.Index(rest, ":")
	if colonIdx < 0 {
		return false
	}
	rest = rest[colonIdx+1:]
	rest = strings.TrimSpace(rest)
	return strings.HasPrefix(rest, "true")
}
