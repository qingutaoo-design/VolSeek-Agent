package memory

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"testing"

	"github.com/joho/godotenv"
	"github.com/pkoukk/tiktoken-go"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/config"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/llm"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

// ============================================================================
// Token 计数（tiktoken cl100k_base）
// ============================================================================

var (
	tokenOnce sync.Once
	tokenTKE  *tiktoken.Tiktoken
	tokenErr  error
)

func getEncoding() (*tiktoken.Tiktoken, error) {
	tokenOnce.Do(func() {
		tokenTKE, tokenErr = tiktoken.GetEncoding("cl100k_base")
	})
	return tokenTKE, tokenErr
}

func countTokens(text string) int {
	tke, err := getEncoding()
	if err != nil {
		return 0
	}
	return len(tke.Encode(text, nil, nil))
}

// ============================================================================
// 模拟摘要函数（快速、可预测，用于纯性能基准测试）
// ============================================================================

// mockSummarizer 截断前 n 字符作为伪摘要，模拟 LLM 压缩
func mockSummarizer(limit int) SummarizeFn {
	return func(_ context.Context, entries []Entry) (string, error) {
		r := []rune(entries[0].Content)
		if len(r) > limit {
			r = r[:limit]
		}
		return "[summary] " + string(r) + "...", nil
	}
}

// say 模拟一轮对话（user + assistant 各一条消息）
func say(mem *BufferMemory, ctx context.Context, sid string, i int) {
	u := fmt.Sprintf("用户第%d条消息：今天天气怎么样？我觉得还行。", i+1)
	a := fmt.Sprintf("第%d轮回答：天气很好，适合出门散步。", i+1)
	mem.Save(ctx, sid, Entry{Role: "user", Content: u, SessionID: sid})
	mem.Save(ctx, sid, Entry{Role: "assistant", Content: a, SessionID: sid})
}

// ============================================================================
// 真实 LLM 摘要函数
// ============================================================================

// newLLMSummarizer 创建使用真实 LLM 进行摘要的 SummarizeFn。
// 从环境变量读取 LLM 配置；若配置缺失则返回 nil。
func newLLMSummarizer(ctx context.Context) SummarizeFn {
	// 加载 .env：尝试当前目录和项目根目录（go test 在包目录运行）
	_ = config.Load()
	_ = godotenv.Load("../../.env")

	apiKey := config.GetEnv("LLM_API_KEY", "")
	baseURL := config.GetEnv("LLM_BASE_URL", "https://api.openai.com/v1")
	model := config.GetEnv("LLM_MODEL", "gpt-4o-mini")
	if apiKey == "" {
		log.Println("[bench] LLM_API_KEY 未配置，无法创建真实 LLM 摘要器")
		return nil
	}

	// Embedding 参数传空值，测试用不到
	client := llm.NewClient(apiKey, baseURL, model, "", "", "")

	return func(ctx context.Context, entries []Entry) (string, error) {
		var lines []string
		for _, e := range entries {
			lines = append(lines, e.Role+": "+e.Content)
		}
		content := strings.Join(lines, "\n")
		prompt := "Summarize the following conversation into 2-3 sentences:\n" + content
		resp, err := client.Chat(ctx, []types.LLMMessage{
			{Role: "user", Content: prompt},
		}, nil, 0.3)
		if err != nil {
			return "", err
		}
		if resp != nil {
			return resp.Content, nil
		}
		return "", nil
	}
}

// ============================================================================
// TestCompressionEfficiency — 验证滑动窗口摘要压缩的 token 节省效果
// 优先使用真实 LLM 进行摘要；若 API 未配置则降级为模拟摘要
// ============================================================================
func TestCompressionEfficiency(t *testing.T) {
	if _, err := getEncoding(); err != nil {
		t.Skipf("tiktoken 初始化失败，跳过: %v", err)
	}

	ctx := context.Background()
	// 10 轮对话（20 条消息），足以触发压缩且避免过多 LLM API 调用
	const rounds = 20

	// 尝试创建真实 LLM 摘要器
	realSum := newLLMSummarizer(ctx)
	if realSum != nil {
		t.Log("使用真实 LLM 进行摘要")
	} else {
		t.Log("LLM_API_KEY 未配置，使用模拟摘要（仅截断）")
	}

	// ---- 基线：不压缩 ----
	t.Log("建立基线（不压缩）...")
	memNo := NewBuffer(200) // 大容量，永不触发压缩
	for i := 0; i < rounds; i++ {
		say(memNo, ctx, "nows", i)
	}
	entries, _ := memNo.Recall(ctx, "nows", 200)
	noTokens := 0
	for _, e := range entries {
		noTokens += countTokens(e.Content)
	}
	t.Logf("无压缩: %d tokens (rounds=%d, 每条消息约 %d tokens)",
		noTokens, rounds, noTokens/(rounds*2))

	// ---- 压缩组：测试不同 windowSize ----
	type result struct {
		windowSize int
		noComp     int
		comp       int
		saved      int
		savedPct   float64
		calls      int
	}

	var results []result
	for _, ws := range []int{2, 4, 6, 8, 10, 12} {
		memComp := NewBuffer(12)
		memComp.SetWindowSize(ws)

		// 选择摘要器：真实 LLM 优先
		summarizer := realSum
		if summarizer == nil {
			summarizer = mockSummarizer(30)
		}

		calls := 0
		origSum := summarizer
		memComp.SetSummarizer(func(ctx context.Context, entries []Entry) (string, error) {
			calls++
			return origSum(ctx, entries)
		})

		for i := 0; i < rounds; i++ {
			say(memComp, ctx, fmt.Sprintf("ws-%d", ws), i)
		}
		entries2, _ := memComp.Recall(ctx, fmt.Sprintf("ws-%d", ws), 200)
		compTokens := 0
		for _, e := range entries2 {
			compTokens += countTokens(e.Content)
		}

		saved := noTokens - compTokens
		pct := 0.0
		if noTokens > 0 {
			pct = float64(saved) / float64(noTokens) * 100
		}
		results = append(results, result{
			windowSize: ws, noComp: noTokens, comp: compTokens,
			saved: saved, savedPct: pct, calls: calls,
		})
		t.Logf("windowSize=%2d  noComp=%5dtok  comp=%5dtok  saved=%5dtok (%5.1f%%)  calls=%d",
			ws, noTokens, compTokens, saved, pct, calls)
	}

	// 找最佳压缩率
	best := results[0]
	for _, r := range results {
		if r.savedPct > best.savedPct {
			best = r
		}
	}
	t.Logf("结论: windowSize=%d 时压缩效率最高 (%.1f%% tokens)", best.windowSize, best.savedPct)

	// 断言：压缩后 token 数必须小于无压缩
	for _, r := range results {
		if r.comp >= r.noComp {
			t.Errorf("windowSize=%d: 压缩后 token 数 (%d) 应小于无压缩 (%d)", r.windowSize, r.comp, r.noComp)
		}
		if r.savedPct <= 0 {
			t.Errorf("windowSize=%d: 压缩率应为正, got %.1f%%", r.windowSize, r.savedPct)
		}
	}
}

// ============================================================================
// 性能基准测试（使用模拟摘要，排除网络延迟干扰）
// ============================================================================

// BenchmarkBufferMemorySave 测量 BufferMemory.Save 在有压缩压力下的性能
func BenchmarkBufferMemorySave(b *testing.B) {
	ctx := context.Background()

	makeEntry := func(role string, i int) Entry {
		return Entry{
			Role:    role,
			Content: fmt.Sprintf("第%04d条消息：今天天气怎么样？我觉得还行，可以去公园散步跑步骑车。", i),
		}
	}

	b.Run("no-compression", func(b *testing.B) {
		mem := NewBuffer(99999)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			mem.Save(ctx, "bench", makeEntry("user", i))
		}
	})

	for _, ws := range []int{2, 4, 6, 8} {
		ws := ws
		b.Run(fmt.Sprintf("windowSize=%d", ws), func(b *testing.B) {
			mem := NewBuffer(12)
			mem.SetWindowSize(ws)
			mem.SetSummarizer(mockSummarizer(30))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				mem.Save(ctx, "bench", makeEntry("user", i))
			}
		})
	}
}

// BenchmarkBufferMemoryRecall 测量 Recall 性能
func BenchmarkBufferMemoryRecall(b *testing.B) {
	ctx := context.Background()
	mem := NewBuffer(200)

	for i := 0; i < 200; i++ {
		mem.Save(ctx, "bench", Entry{
			Role:    "user",
			Content: fmt.Sprintf("第%04d条消息：今天天气怎么样？我觉得还行。", i),
		})
	}

	for _, limit := range []int{5, 10, 20, 50} {
		limit := limit
		b.Run(fmt.Sprintf("limit=%d", limit), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				mem.Recall(ctx, "bench", limit)
			}
		})
	}
}
