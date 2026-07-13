package initapp

import (
	"context"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/agent"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/rag"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

func NewRouter(volseek *agent.AgentEngine, store rag.Store, graphStore *rag.GraphStore) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type")
		if c.Request.Method == "OPTIONS" { c.AbortWithStatus(204); return }
		c.Next()
	})
	r.GET("/api/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok", "chunks": store.Len(), "time": time.Now().Format(time.RFC3339)})
	})
	r.GET("/api/stats", func(c *gin.Context) {
		e, r2 := graphStore.Stats()
		c.JSON(200, gin.H{"chunks": store.Len(), "entities": e, "relations": r2})
	})
	r.POST("/api/query", func(c *gin.Context) {
		var req struct{ Query string `json:"query" binding:"required"` }
		if err := c.ShouldBindJSON(&req); err != nil { c.JSON(400, gin.H{"error": "missing query"}); return }
		c.Header("Content-Type", "text/event-stream"); c.Header("Cache-Control", "no-cache"); c.Header("Connection", "keep-alive")
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		eventCh, err := volseek.Execute(ctx, req.Query)
		if err != nil { c.SSEvent("error", gin.H{"message": err.Error()}); c.Writer.Flush(); return }
		for event := range eventCh { c.SSEvent("message", gin.H{"type": event.Type, "content": event.Content, "done": event.Done}); c.Writer.Flush(); if event.Done { break } }
	})
	r.POST("/api/query/sync", func(c *gin.Context) {
		var req struct{ Query string `json:"query" binding:"required"` }
		if err := c.ShouldBindJSON(&req); err != nil { c.JSON(400, gin.H{"error": "missing query"}); return }
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		eventCh, err := volseek.Execute(ctx, req.Query)
		if err != nil { c.JSON(500, gin.H{"error": err.Error()}); return }
		var answer string; var conf float64
		for event := range eventCh {
			switch event.Type {
			case types.EventAnswer: answer += event.Content
			case types.EventDone: if d, ok := event.Data.(map[string]interface{}); ok { if c, ok := d["confidence"].(float64); ok { conf = c } }
			}
		}
		c.JSON(200, gin.H{"answer": answer, "confidence": conf})
	})
	r.POST("/api/chat", func(c *gin.Context) {
		var req struct{ Query string `json:"query" binding:"required"`; SessionID string `json:"session_id"` }
		if err := c.ShouldBindJSON(&req); err != nil { c.JSON(400, gin.H{"error": "missing query"}); return }
		if req.SessionID == "" { req.SessionID = "default" }
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		memCtx := context.WithValue(ctx, "session_id", req.SessionID)
		eventCh, err := volseek.Execute(memCtx, req.Query)
		if err != nil { c.JSON(500, gin.H{"error": err.Error()}); return }
		var answer string; var conf float64
		for event := range eventCh {
			switch event.Type {
			case types.EventAnswer: answer += event.Content
			case types.EventDone: if d, ok := event.Data.(map[string]interface{}); ok { if c, ok := d["confidence"].(float64); ok { conf = c } }
			}
		}
		c.JSON(200, gin.H{"answer": answer, "confidence": conf, "session_id": req.SessionID})
	})
	return r
}
