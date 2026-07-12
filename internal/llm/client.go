// Package llm 提供统一的 LLM 调用接口，支持 OpenAI/DeepSeek 等兼容 API。
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

// ChatResponse 是 LLM 非流式调用的响应。
type ChatResponse struct {
	Content      string
	ToolCalls    []types.ToolCallDef
	FinishReason string
}

// Client 封装了 LLM 的所有调用能力。
type Client struct {
	client      *openai.Client
	model       string
	embedClient *openai.Client // 独立的 Embedding 客户端（不同服务商时使用）
	embedModel  string         // Embedding 模型名
}

// NewClient 创建 LLM 客户端。
// embedBaseURL / embedModel / embedAPIKey 为空时，Embedding 复用聊天配置。
func NewClient(apiKey, baseURL, model, embedBaseURL, embedModel, embedAPIKey string) *Client {
	if apiKey == "" {
		apiKey = "sk-placeholder"
	}
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	if model == "" {
		model = "gpt-4o-mini"
	}
	if embedModel == "" {
		embedModel = "text-embedding-ada-002"
	}
	if embedAPIKey == "" {
		embedAPIKey = apiKey // 没配独立密钥时，复用聊天密钥
	}
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = baseURL
	c := &Client{
		client:     openai.NewClientWithConfig(cfg),
		model:      model,
		embedModel: embedModel,
	}

	// 如果提供了独立的 Embedding 服务地址，创建专用客户端
	if embedBaseURL != "" && embedBaseURL != baseURL {
		eCfg := openai.DefaultConfig(embedAPIKey)
		eCfg.BaseURL = embedBaseURL
		c.embedClient = openai.NewClientWithConfig(eCfg)
	} else {
		c.embedClient = c.client
	}

	return c
}

// Chat 执行非流式聊天补全。
func (c *Client) Chat(ctx context.Context, messages []types.LLMMessage, tools []openai.Tool, temperature float64) (*ChatResponse, error) {
	openaiMsgs := c.toOpenAIMessages(messages)

	req := openai.ChatCompletionRequest{
		Model:       c.model,
		Messages:    openaiMsgs,
		Temperature: float32(temperature),
	}
	if len(tools) > 0 {
		req.Tools = tools
	}

	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("LLM chat failed: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("LLM returned no choices")
	}

	choice := resp.Choices[0]
	result := &ChatResponse{
		Content:      choice.Message.Content,
		FinishReason: string(choice.FinishReason),
	}

	for _, tc := range choice.Message.ToolCalls {
		result.ToolCalls = append(result.ToolCalls, types.ToolCallDef{
			ID:        tc.ID,
			Type:      string(tc.Type),
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}

	return result, nil
}

// ChatStream 执行流式聊天补全。
func (c *Client) ChatStream(ctx context.Context, messages []types.LLMMessage, tools []openai.Tool, temperature float64) (<-chan types.StreamEvent, error) {
	openaiMsgs := c.toOpenAIMessages(messages)

	req := openai.ChatCompletionRequest{
		Model:       c.model,
		Messages:    openaiMsgs,
		Temperature: float32(temperature),
		Stream:      true,
	}
	if len(tools) > 0 {
		req.Tools = tools
	}

	stream, err := c.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("LLM stream failed: %w", err)
	}

	ch := make(chan types.StreamEvent, 64)

	go func() {
		defer close(ch)
		defer stream.Close()

		var (
			fullContent string
			toolCalls   []toolCallAccumulator
		)

		for {
			resp, err := stream.Recv()
			if err != nil {
				if err == io.EOF {
					ch <- types.StreamEvent{Type: types.EventDone, Content: fullContent, Done: true}
					return
				}
				ch <- types.StreamEvent{Type: types.EventError, Content: fmt.Sprintf("stream error: %v", err), Done: true}
				return
			}

			if len(resp.Choices) == 0 {
				continue
			}

			delta := resp.Choices[0].Delta

			if delta.Content != "" {
				fullContent += delta.Content
				ch <- types.StreamEvent{Type: types.EventAnswer, Content: delta.Content}
			}

			if len(delta.ToolCalls) > 0 {
				for _, tc := range delta.ToolCalls {
					if tc.Index != nil {
						idx := *tc.Index
						for idx >= len(toolCalls) {
							toolCalls = append(toolCalls, toolCallAccumulator{})
						}
						if tc.Function.Name != "" {
							toolCalls[idx].Name = tc.Function.Name
						}
						if tc.ID != "" {
							toolCalls[idx].ID = tc.ID
						}
						toolCalls[idx].Arguments += tc.Function.Arguments
					}
				}
			}

			if resp.Choices[0].FinishReason != "" {
				for _, tc := range toolCalls {
					ch <- types.StreamEvent{
						Type: types.EventToolCall,
						Data: types.ToolCallDef{
							ID:        tc.ID,
							Name:      tc.Name,
							Arguments: tc.Arguments,
						},
					}
				}
				ch <- types.StreamEvent{Type: types.EventDone, Content: fullContent, Done: true}
			}
		}
	}()

	return ch, nil
}

// toolCallAccumulator 用于累积流式 tool call 的增量数据。
type toolCallAccumulator struct {
	ID        string
	Name      string
	Arguments string
}

// Embed 将文本向量化。
func (c *Client) Embed(ctx context.Context, text string) ([]float64, error) {
	ec := c.embedClient
	if ec == nil {
		ec = c.client
	}
	resp, err := ec.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: openai.EmbeddingModel(c.embedModel),
		Input: []string{text},
	})
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("embedding returned no data")
	}
	// OpenAI 返回 float32，转为 float64
	return float32to64(resp.Data[0].Embedding), nil
}

// EmbedBatch 批量向量化文本。
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	const batchSize = 10 // DashScope 限制每批≤10，OpenAI 支持更高
	results := make([][]float64, len(texts))

	ec := c.embedClient
	if ec == nil {
		ec = c.client
	}

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]
		resp, err := ec.CreateEmbeddings(ctx, openai.EmbeddingRequest{
			Model: openai.EmbeddingModel(c.embedModel),
			Input: batch,
		})
		if err != nil {
			return nil, fmt.Errorf("batch embedding failed at offset %d: %w", i, err)
		}
		for _, data := range resp.Data {
			if data.Index < len(batch) {
				results[i+data.Index] = float32to64(data.Embedding)
			}
		}
		if end < len(texts) {
			time.Sleep(200 * time.Millisecond)
		}
	}
	return results, nil
}

// ChatWithRetry 带重试的聊天补全。
func (c *Client) ChatWithRetry(ctx context.Context, messages []types.LLMMessage, tools []openai.Tool, temperature float64) (*ChatResponse, error) {
	maxRetries := 3
	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			log.Printf("[LLM] Retry %d/%d: %v", attempt+1, maxRetries, lastErr)
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
		result, err := c.Chat(ctx, messages, tools, temperature)
		if err == nil {
			return result, nil
		}
		lastErr = err
		if !isRetryable(err) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("LLM call failed after retries: %w", lastErr)
}

// toOpenAIMessages 将内部消息转为 OpenAI 格式。
func (c *Client) toOpenAIMessages(msgs []types.LLMMessage) []openai.ChatCompletionMessage {
	result := make([]openai.ChatCompletionMessage, 0, len(msgs))
	for _, msg := range msgs {
		oaiMsg := openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
		if len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				oaiMsg.ToolCalls = append(oaiMsg.ToolCalls, openai.ToolCall{
					ID:   tc.ID,
					Type: openai.ToolType(tc.Type),
					Function: openai.FunctionCall{
						Name:      tc.Name,
						Arguments: tc.Arguments,
					},
				})
			}
		}
		if msg.ToolCallID != "" {
			oaiMsg.ToolCallID = msg.ToolCallID
		}
		result = append(result, oaiMsg)
	}
	return result
}

// ToolToOpenAI 将内部工具定义转为 OpenAI Tool 格式。
func ToolToOpenAI(name, description string, parameters json.RawMessage) openai.Tool {
	return openai.Tool{
		Type: "function",
		Function: &openai.FunctionDefinition{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}

// float32to64 将 float32 切片转为 float64。
func float32to64(src []float32) []float64 {
	dst := make([]float64, len(src))
	for i, v := range src {
		dst[i] = float64(v)
	}
	return dst
}

// isRetryable 判断错误是否可重试。
func isRetryable(err error) bool {
	errStr := strings.ToLower(err.Error())
	markers := []string{"429", "500", "502", "503", "timeout", "rate limit", "temporarily unavailable", "server error"}
	for _, m := range markers {
		if strings.Contains(errStr, m) {
			return true
		}
	}
	return false
}
