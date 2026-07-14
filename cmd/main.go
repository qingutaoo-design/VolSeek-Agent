// VolSeek-Agent — HTTP API server for RAG Agent.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/qingutaoo-design/VolSeek-Agent/internal/config"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/initapp"
)

func main() {
	if err := config.Load(); err != nil {
		log.Printf("Warning: config load: %v", err)
	}

	volseek, store, graphStore, _, kbManager := initapp.InitAgent(context.Background())

	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("🤖 VolSeek-Agent 已就绪！")
	fmt.Println(strings.Repeat("=", 60))

	router := initapp.NewRouter(volseek, store, graphStore, kbManager)
	srv := &http.Server{Addr: ":8080", Handler: router}

	go func() {
		log.Printf("🌐 Frontend: http://localhost%s", srv.Addr)
		log.Printf("🌐 API:     http://localhost%s/api/health", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("API server failed: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	log.Printf("🛑 Received %v, shutting down...", sig)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	srv.Shutdown(ctx)
	log.Println("👋 Exited gracefully")
}
