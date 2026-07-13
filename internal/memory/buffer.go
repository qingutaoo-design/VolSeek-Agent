package memory

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// SummarizeFn 将一批旧对话浓缩成一条摘要。agent 调用方注入 LLM 实现。
type SummarizeFn func(ctx context.Context, entries []Entry) (summary string, err error)

// BufferMemory 情节记忆：滑动窗口 + 摘要压缩。
// 超过容量时，最旧的一批消息被 LLM 压缩成一条摘要，而非直接丢弃。
type BufferMemory struct {
	mu           sync.RWMutex
	capacity     int               // 最大消息条数
	windowSize   int               // 每次压缩多少条
	entries      map[string][]Entry
	summarizer   SummarizeFn       // 可选：LLM 摘要函数，nil 时直接丢弃
}

// NewBuffer 创建 BufferMemory。capacity 为窗口上限，默认 50。
func NewBuffer(capacity int) *BufferMemory {
	if capacity <= 0 { capacity = 50 }
	return &BufferMemory{
		capacity:   capacity,
		windowSize: 4, // 每次压缩 4 条（2 轮对话）
		entries:    make(map[string][]Entry),
	}
}

// SetSummarizer 注入 LLM 摘要函数。不设置时超出容量直接丢弃最旧消息。
func (bm *BufferMemory) SetSummarizer(fn SummarizeFn) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	bm.summarizer = fn
}

// SetWindowSize 设置每次压缩的条目数（必须是偶数，默认 4）。
func (bm *BufferMemory) SetWindowSize(n int) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	if n < 2 { n = 2 }
	if n%2 != 0 { n++ } // 确保偶数
	bm.windowSize = n
}

func (bm *BufferMemory) Save(ctx context.Context, sessionID string, entry Entry) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	bm.entries[sessionID] = append(bm.entries[sessionID], entry)

	// 未超容量 → 直接返回
	if len(bm.entries[sessionID]) <= bm.capacity {
		return nil
	}

	// 超容量 → 压缩最旧的 windowSize 条，或直接丢弃
	entries := bm.entries[sessionID]
	old := entries[:bm.windowSize]
	rest := entries[bm.windowSize:]

	if bm.summarizer != nil {
		summary, err := bm.summarizer(ctx, old)
		if err == nil && summary != "" {
			// 用一条摘要条目替换被压缩的旧消息
			rest = append([]Entry{{
				Role: "system", Content: summary,
				Summary: "compressed",
			}}, rest...)
		}
		// 摘要失败时丢弃旧消息（跟没 summarizer 一样）
	}

	bm.entries[sessionID] = rest
	return nil
}

func (bm *BufferMemory) Recall(_ context.Context, sessionID string, limit int) ([]Entry, error) {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	entries := bm.entries[sessionID]
	if len(entries) == 0 { return nil, nil }
	if limit <= 0 || limit > len(entries) { limit = len(entries) }

	result := make([]Entry, limit)
	copy(result, entries[len(entries)-limit:])
	return result, nil
}

func (bm *BufferMemory) Clear(_ context.Context, sessionID string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	delete(bm.entries, sessionID)
	return nil
}

// BuildContext 将记忆条目拼成 system prompt 格式的上下文文本。
func BuildContext(entries []Entry) string {
	if len(entries) == 0 { return "" }
	var sb strings.Builder
	sb.WriteString("\n## Conversation History\n")
	for _, e := range entries {
		prefix := ""
		switch e.Role {
		case "assistant": prefix = "you"
		case "system":    prefix = "[summary]"
		default:          prefix = e.Role
		}
		sb.WriteString(fmt.Sprintf("%s: %s\n", prefix, e.Content))
	}
	return sb.String()
}
