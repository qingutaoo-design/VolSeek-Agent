package rag

import (
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

// ============================================================
// 样本数据
// ============================================================

// 纯文本样本 — 长段落，无明显结构
const samplePlainText = `Go语言是由Google开发的一种静态强类型、编译型、并发型，并具有垃圾回收功能的编程语言。

Robert Griesemer、Rob Pike和Ken Thompson在2007年设计了Go语言，并于2009年11月正式对外发布。Go语言借鉴了C语言的语法结构，同时吸收了Pascal、Modula等语言的优点。

Go语言的主要特点包括：简洁的语法、高效的编译速度、内置的并发支持、丰富的标准库、垃圾回收机制、跨平台支持等。其中，goroutine和channel是Go语言并发编程的核心机制。goroutine是一种轻量级线程，由Go运行时管理，创建成本极低。channel则用于goroutine之间的通信，遵循"不要通过共享内存来通信，而应该通过通信来共享内存"的设计哲学。

Go语言在云计算、微服务、网络编程、DevOps等领域得到了广泛应用。Docker、Kubernetes、Prometheus、Etcd等知名项目都是用Go语言编写的。这些项目的成功证明了Go语言在构建高性能、可扩展的系统方面具有显著优势。

Go语言的工具链也非常完善，包括格式化工具gofmt、依赖管理工具go mod、测试工具go test、性能分析工具pprof等。这些工具大大提高了开发效率和代码质量。`

// 结构化工整文本 — 段落清晰、句号完整
const sampleStructuredText = `机器学习是人工智能的一个分支。
它主要研究如何让计算机从数据中自动学习规律和模式。
监督学习是最常见的机器学习范式。
它使用标注数据来训练模型。
无监督学习则使用未标注数据。
它试图发现数据中的隐藏结构。
强化学习让智能体通过与环境交互来学习策略。
深度学习是机器学习的一个重要分支。
它使用多层神经网络来学习数据的层次化表示。
卷积神经网络擅长处理图像数据。
循环神经网络适合处理序列数据。
Transformer架构彻底改变了自然语言处理领域。
BERT和GPT是两种著名的Transformer模型。
BERT使用双向编码器来理解上下文。
GPT使用单向解码器来生成文本。
这些模型在各种NLP任务上都取得了突破性进展。`

// Markdown 样本 — 多层标题，代码块，列表
const sampleMarkdown = `# Go语言并发编程指南

## 概述

Go语言在语言层面原生支持并发，这是它区别于其他编程语言的重要特性。
Go的并发模型基于CSP（Communicating Sequential Processes）理论。

## Goroutine

### 什么是Goroutine

Goroutine是Go语言中的轻量级线程，由Go运行时（runtime）管理。
与操作系统线程相比，goroutine的创建和销毁成本非常低，初始栈空间只有几KB。

### 启动Goroutine

使用go关键字即可启动一个goroutine：

` + "```go" + `
go func() {
    fmt.Println("Hello from goroutine")
}()
` + "```" + `

## Channel

### Channel的类型

Channel分为无缓冲channel和有缓冲channel两种类型。
无缓冲channel要求发送和接收操作同时准备好，否则会阻塞。
有缓冲channel在缓冲区未满时可以异步发送。

### 使用Channel通信

` + "```go" + `
ch := make(chan int)
go func() {
    ch <- 42  // 发送数据
}()
value := <-ch  // 接收数据
fmt.Println(value)
` + "```" + `

## 同步原语

### WaitGroup

sync.WaitGroup用于等待一组goroutine完成任务。
它提供了Add、Done和Wait三个方法。

### Mutex

sync.Mutex用于保护共享资源的互斥访问。
它提供了Lock和Unlock方法。

## 最佳实践

1. 不要通过共享内存来通信，而应该通过通信来共享内存
2. 尽量使用channel来传递数据，减少共享变量的使用
3. 使用context包来处理超时和取消信号
4. 避免死锁和活锁的正确使用锁和channel`

// ============================================================
// 测试：两种策略直接对比
// ============================================================

func TestCompareStrats_PlainText(t *testing.T) {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("测试 1：纯文本（长段落）")
	fmt.Println(strings.Repeat("=", 70))

	cfg := types.ChunkConfig{Size: 150, Overlap: 30}
	chunker := NewChunker(&cfg)

	// 强制固定大小分块（因为纯文本不会触发 Markdown 策略）
	fixedChunks := chunker.chunkByFixedSize(samplePlainText, "纯文本样本")

	// 强制 Markdown 分块（看看它对纯文本的表现）
	mdChunks := chunker.chunkByMarkdown(samplePlainText, "纯文本样本")

	printComparison(t, "固定大小分块", fixedChunks, cfg)
	printComparison(t, "Markdown段落分块", mdChunks, cfg)
}

func TestCompareStrats_StructuredText(t *testing.T) {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("测试 2：结构化工整文本（短句 + 句号结尾）")
	fmt.Println(strings.Repeat("=", 70))

	cfg := types.ChunkConfig{Size: 100, Overlap: 20}
	chunker := NewChunker(&cfg)

	fixedChunks := chunker.chunkByFixedSize(sampleStructuredText, "结构化文本")
	mdChunks := chunker.chunkByMarkdown(sampleStructuredText, "结构化文本")

	printComparison(t, "固定大小分块", fixedChunks, cfg)
	printComparison(t, "Markdown段落分块", mdChunks, cfg)
}

func TestCompareStrats_Markdown(t *testing.T) {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("测试 3：Markdown 文档（多层标题 + 代码块）")
	fmt.Println(strings.Repeat("=", 70))

	cfg := types.ChunkConfig{Size: 150, Overlap: 30}
	chunker := NewChunker(&cfg)

	// 自动策略（Chunk 会检测 Markdown 走段落分块）
	autoChunks := chunker.Chunk(sampleMarkdown, "Go并发编程")

	// 强制固定大小分块
	fixedChunks := chunker.chunkByFixedSize(sampleMarkdown, "Go并发编程")

	printComparison(t, "自动策略（Markdown段落分块）", autoChunks, cfg)
	printComparison(t, "固定大小分块", fixedChunks, cfg)
}

func TestCompareStrats_ShortText(t *testing.T) {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("测试 4：短文本（小于 Size）")
	fmt.Println(strings.Repeat("=", 70))

	short := "Go语言是一种静态强类型的编程语言。"

	cfg := types.ChunkConfig{Size: 500, Overlap: 50}
	chunker := NewChunker(&cfg)

	fixedChunks := chunker.chunkByFixedSize(short, "短文本")
	mdChunks := chunker.chunkByMarkdown(short, "短文本")

	printComparison(t, "固定大小分块", fixedChunks, cfg)
	printComparison(t, "Markdown段落分块", mdChunks, cfg)
}

// ============================================================
// 辅助：打印对比报告
// ============================================================

func printComparison(t *testing.T, label string, chunks []*types.Chunk, cfg types.ChunkConfig) {
	fmt.Printf("\n【%s】\n", label)
	fmt.Printf("  配置: Size=%d, Overlap=%d\n", cfg.Size, cfg.Overlap)
	fmt.Printf("  产出块数: %d\n", len(chunks))

	if len(chunks) == 0 {
		fmt.Println("  (无输出)")
		return
	}

	// 统计指标
	var totalRunes int
	var maxRunes int
	var minRunes = int(^uint(0) >> 1) // MaxInt
	sentenceBroken := 0
	headingPreserved := 0

	for i, ch := range chunks {
		runeCount := utf8.RuneCountInString(ch.Content)
		totalRunes += runeCount
		if runeCount > maxRunes {
			maxRunes = runeCount
		}
		if runeCount < minRunes {
			minRunes = runeCount
		}

		// 检测块末尾是否在句子边界（用 rune 处理中文标点）
		trimmed := strings.TrimSpace(ch.Content)
		if len(trimmed) > 0 {
			trimmedRunes := []rune(trimmed)
			lastRune := trimmedRunes[len(trimmedRunes)-1]
			if lastRune != '.' && lastRune != '。' && lastRune != '！' && lastRune != '？' &&
				lastRune != '\n' && lastRune != '`' {
				sentenceBroken++
			}
		}

		// 检测是否保留了标题信息
		if strings.Contains(ch.DocTitle, ">") {
			headingPreserved++
		}

		// 打印块内容（截断显示）
		display := ch.Content
		if runeCount > 80 {
			runes := []rune(display)
			display = string(runes[:40]) + " ... " + string(runes[len(runes)-30:])
		}
		fmt.Printf("  [块 %2d | %3d runes | DocTitle: %s]\n    %s\n",
			i, runeCount, ch.DocTitle, strings.ReplaceAll(display, "\n", "\\n"))
	}

	avg := totalRunes / len(chunks)
	fmt.Printf("\n  统计:\n")
	fmt.Printf("    平均块大小: %d runes\n", avg)
	fmt.Printf("    最小块: %d runes, 最大块: %d runes\n", minRunes, maxRunes)
	fmt.Printf("    块大小波动: ±%d runes (%.1f%%)\n",
		maxRunes-minRunes, float64(maxRunes-minRunes)/float64(avg)*100)
	fmt.Printf("    非句子末尾截断: %d/%d (%.0f%%)\n",
		sentenceBroken, len(chunks), float64(sentenceBroken)/float64(len(chunks))*100)
	if headingPreserved > 0 {
		fmt.Printf("    保留标题层级的块: %d/%d\n", headingPreserved, len(chunks))
	}
	fmt.Println()
}

// ============================================================
// 测试：自动路由检测准确性
// ============================================================

func TestChunkRouteDetection(t *testing.T) {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("测试 5：Chunk() 自动路由检测")
	fmt.Println(strings.Repeat("=", 70))

	tests := []struct {
		name string
		text string
	}{
		{"纯文本（无标题标记）", samplePlainText},
		{"Markdown（有#标题）", sampleMarkdown},
		{"含##但不含#", "这是一个##测试\n但第一行不是标题"},
		{"普通文本含#号", "C#是一种编程语言\n#标签不是标题"},
	}

	cfg := types.ChunkConfig{Size: 200, Overlap: 30}
	chunker := NewChunker(&cfg)

	for _, tt := range tests {
		chunks := chunker.Chunk(tt.text, "测试")

		// 判断实际走了哪条路径
		// 如果 chunk 的 DocTitle 包含 ">" 说明走了 Markdown 分支
		usedMarkdown := false
		for _, ch := range chunks {
			if strings.Contains(ch.DocTitle, ">") {
				usedMarkdown = true
				break
			}
		}
		route := "固定大小分块"
		if usedMarkdown {
			route = "Markdown段落分块"
		} else if strings.Contains(tt.text, "# ") || strings.Contains(tt.text, "## ") {
			route = "❌ 含#标记但走了固定分块（可能误判）"
		}
		fmt.Printf("  %-30s -> %s (%d chunks)\n", tt.name, route, len(chunks))
	}
	fmt.Println()
}

// ============================================================
// 测试：边界情况
// ============================================================

func TestChunkerEdgeCases(t *testing.T) {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("测试 6：边界情况")
	fmt.Println(strings.Repeat("=", 70))

	cfg := types.ChunkConfig{Size: 100, Overlap: 20}
	chunker := NewChunker(&cfg)

	// 空文本
	empty := chunker.Chunk("", "空文档")
	fmt.Printf("  空文本: %d chunks\n", len(empty))

	// 刚好等于 Size
	exact := chunker.chunkByFixedSize(strings.Repeat("a", 100), "恰好100")
	fmt.Printf("  恰好Size(100): %d chunks\n", len(exact))

	// 略大于 Size
	slightly := chunker.chunkByFixedSize(strings.Repeat("a", 101), "略大于100")
	fmt.Printf("  略大于Size(101): %d chunks\n", len(slightly))

	// 只含标题的文本
	headings := "# 标题一\n\n## 标题二\n\n### 标题三\n"
	hdChunks := chunker.Chunk(headings, "仅标题")
	fmt.Printf("  仅标题文本: %d chunks\n", len(hdChunks))
	for _, ch := range hdChunks {
		fmt.Printf("    [DocTitle: %s] len=%d\n", ch.DocTitle, utf8.RuneCountInString(ch.Content))
	}

	fmt.Println()
}
