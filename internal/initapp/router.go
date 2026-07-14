package initapp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/agent"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/knowledge"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/rag"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

func NewRouter(volseek *agent.AgentEngine, store rag.Store, graphStore *rag.GraphStore, kbManager *knowledge.Manager) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	r.MaxMultipartMemory = 100 << 20 // 100MB 最大上传
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type,Authorization")
		if c.Request.Method == "OPTIONS" { c.AbortWithStatus(204); return }
		c.Next()
	})
	// Serve static frontend files at root paths (matching HTML relative references)
	r.Static("/css", "./frontend/css")
	r.Static("/js", "./frontend/js")
	r.StaticFile("/", "./frontend/index.html")
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
		c.Header("Content-Type", "text/event-stream;charset=utf-8"); c.Header("Cache-Control", "no-cache"); c.Header("Connection", "keep-alive")
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		eventCh, err := volseek.Execute(ctx, req.Query)
		if err != nil {
			data, _ := json.Marshal(gin.H{"message": err.Error()})
			fmt.Fprintf(c.Writer, "event: error\ndata: %s\n\n", data)
			c.Writer.Flush()
			return
		}
		for event := range eventCh {
			data, _ := json.Marshal(gin.H{"type": event.Type, "content": event.Content, "done": event.Done})
			fmt.Fprintf(c.Writer, "event: message\ndata: %s\n\n", data)
			c.Writer.Flush()
			if event.Done { break }
		}
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
	// ============================================================
	// 知识库管理 API
	// ============================================================
	if kbManager != nil {
		kbGroup := r.Group("/api/knowledge")
		{
			// 上传文件内容（multipart/form-data）
			kbGroup.POST("/upload", func(c *gin.Context) {
				file, err := c.FormFile("file")
				if err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": "missing file field — 请使用 multipart/form-data 的 file 字段上传文件内容"})
					return
				}
				f, err := file.Open()
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{"error": "cannot open uploaded file"})
					return
				}
				defer f.Close()

				ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
				defer cancel()

				fi, err := kbManager.Upload(ctx, file.Filename, f)
				if err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
					return
				}
				c.JSON(http.StatusOK, gin.H{
					"uuid":        fi.UUID,
					"name":        fi.Name,
					"size":        fi.Size,
					"chunk_count": fi.ChunkCount,
					"created_at":  fi.CreatedAt.Format(time.RFC3339),
				})
			})

			// 列出所有文件
			kbGroup.GET("/files", func(c *gin.Context) {
				files := kbManager.ListFiles()
				type fileResp struct {
					UUID       string `json:"uuid"`
					Name       string `json:"name"`
					Size       int64  `json:"size"`
					ChunkCount int    `json:"chunk_count"`
					CreatedAt  string `json:"created_at"`
				}
				resp := make([]fileResp, 0, len(files))
				for _, f := range files {
					resp = append(resp, fileResp{
						UUID:       f.UUID,
						Name:       f.Name,
						Size:       f.Size,
						ChunkCount: f.ChunkCount,
						CreatedAt:  f.CreatedAt.Format(time.RFC3339),
					})
				}
				c.JSON(http.StatusOK, gin.H{"files": resp, "total": len(files)})
			})

			// 删除文件
			kbGroup.DELETE("/files/:uuid", func(c *gin.Context) {
				uuid := c.Param("uuid")
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				defer cancel()

				if err := kbManager.DeleteFile(ctx, uuid); err != nil {
					c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
					return
				}
				c.JSON(http.StatusOK, gin.H{"message": "file deleted"})
			})

			// 从本地路径导入文件（传路径，服务端自己读）
			kbGroup.POST("/import", func(c *gin.Context) {
				var req struct {
					Path string `json:"path" binding:"required"`
				}
				if err := c.ShouldBindJSON(&req); err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": "missing path — 请提供本地文件路径"})
					return
				}

				// 打开本地文件
				f, err := os.Open(req.Path)
				if err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": "cannot open file: " + err.Error()})
					return
				}
				defer f.Close()

				// 提取文件名
				name := req.Path
				if idx := strings.LastIndexAny(name, "\\/"); idx >= 0 {
					name = name[idx+1:]
				}

				ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
				defer cancel()

				fi, err := kbManager.Upload(ctx, name, f)
				if err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
					return
				}
				c.JSON(http.StatusOK, gin.H{
					"uuid":        fi.UUID,
					"name":        fi.Name,
					"size":        fi.Size,
					"chunk_count": fi.ChunkCount,
					"created_at":  fi.CreatedAt.Format(time.RFC3339),
				})
			})

			// 重新索引所有文件
			kbGroup.POST("/reindex", func(c *gin.Context) {
				ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
				defer cancel()

				fileCount, totalChunks, err := kbManager.ReindexAll(ctx)
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
					return
				}
				c.JSON(http.StatusOK, gin.H{
					"message":      "reindex complete",
					"file_count":   fileCount,
					"total_chunks": totalChunks,
				})
			})
		}
	}

	return r
}
