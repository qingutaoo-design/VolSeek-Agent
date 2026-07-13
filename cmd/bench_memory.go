//go:build ignore
// Benchmark: 验证滑动窗口摘要压缩的 token 节省效果
// 运行: go run cmd/bench_memory.go
package main

import (
	"context"
	"fmt"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/memory"
)

type benchResult struct {
	windowSize   int
	totalRounds  int
	noComp       int
	comp         int
	saved        int
	savedPercent float64
	calls        int
}

func main() {
	fmt.Println("=== BufferMemory 压缩效率 Benchmark ===")
	fmt.Println("模拟 30 轮对话 (60 条消息), capacity=12, 摘要模拟为 30 字截断\n")

	var results []benchResult
	for _, ws := range []int{2, 4, 6, 8} {
		r := run(ws)
		results = append(results, r)
		fmt.Printf("windowSize=%2d  noComp=%5d  comp=%5d  saved=%5d (%5.1f%%)  calls=%d\n",
			r.windowSize, r.noComp, r.comp, r.saved, r.savedPercent, r.calls)
	}

	best := results[0]
	for _, r := range results {
		if r.savedPercent > best.savedPercent {
			best = r
		}
	}
	fmt.Printf("\n结论: windowSize=%d 时压缩效率最高 (%.1f%%)\n", best.windowSize, best.savedPercent)
	fmt.Println("建议默认 windowSize=4，平衡压缩率与摘要精度")
}

func run(ws int) benchResult {
	ctx := context.Background()
	const rounds = 30

	// 不压缩组：大容量 buffer，始终不触发压缩
	memNo := memory.NewBuffer(200)
	for i := 0; i < rounds; i++ {
		say(memNo, ctx, "nows", i)
	}
	entries, _ := memNo.Recall(ctx, "nows", 200)
	noChars := 0
	for _, e := range entries {
		noChars += len([]rune(e.Content))
	}

	// 压缩组：小容量 buffer，频繁触发压缩
	memComp := memory.NewBuffer(12)
	memComp.SetWindowSize(ws)
	calls := 0
	memComp.SetSummarizer(func(_ context.Context, entries []memory.Entry) (string, error) {
		calls++
		r := []rune(entries[0].Content)
		if len(r) > 30 {
			r = r[:30]
		}
		return "[summary] " + string(r) + "...", nil
	})
	for i := 0; i < rounds; i++ {
		say(memComp, ctx, "ws", i)
	}
	entries2, _ := memComp.Recall(ctx, "ws", 200)
	compChars := 0
	for _, e := range entries2 {
		compChars += len([]rune(e.Content))
	}

	saved := noChars - compChars
	pct := 0.0
	if noChars > 0 {
		pct = float64(saved) / float64(noChars) * 100
	}
	return benchResult{
		windowSize: ws, totalRounds: rounds,
		noComp: noChars, comp: compChars,
		saved: saved, savedPercent: pct, calls: calls,
	}
}

func say(mem *memory.BufferMemory, ctx context.Context, sid string, i int) {
	u := fmt.Sprintf("用户第%d条消息：今天天气怎么样？我觉得还行。", i+1)
	a := fmt.Sprintf("第%d轮回答：天气很好，适合出门散步。", i+1)
	mem.Save(ctx, sid, memory.Entry{Role: "user", Content: u, SessionID: sid})
	mem.Save(ctx, sid, memory.Entry{Role: "assistant", Content: a, SessionID: sid})
}
