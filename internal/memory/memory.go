// Package memory 提供两级记忆系统：工作记忆 / 情节记忆
package memory

import (
	"context"
	"time"
)

// Entry 表示一条记忆记录。
type Entry struct {
	Role      string    `json:"role"`    // "user" | "assistant" | "system"
	Content   string    `json:"content"` // 消息原文
	Summary   string    `json:"summary"` // LLM 生成的摘要（情节记忆用）
	SessionID string    `json:"session_id"`
	Timestamp time.Time `json:"timestamp"`
}

// Memory 是记忆系统的核心接口。
type Memory interface {
	// Save 保存一条记忆。
	Save(ctx context.Context, sessionID string, entry Entry) error

	// Recall 根据查询召回相关记忆，返回按时间降序排列的记录。
	Recall(ctx context.Context, sessionID string, limit int) ([]Entry, error)

	// Clear 清空指定会话的记忆。
	Clear(ctx context.Context, sessionID string) error
}

// 记忆三级体系：
//
// 1. 工作记忆（WorkingMemory）
//    当前会话的消息列表，存储在 LLM 的 context window 中。
//    由 agent 的 messages []LLMMessage 管理，本包不额外存储。
//
// 2. 情节记忆（BufferMemory）
//    跨轮次的消息摘要，存储最近 N 轮对话的要点。
//    用环形缓冲区实现，内存存储，会话级。
