// Package knowledge 提供本地知识库管理能力。
// 支持文件上传、删除、重新索引，以及启动时自动扫描加载。
// 文件存储在 KB_DIR 指定的本地目录中，Agent 通过 RAG 引擎
// 自动识别并索引这些文件为可检索的向量分块。
package knowledge

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/llm"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/rag"
	"github.com/qingutaoo-design/VolSeek-Agent/internal/types"
)

// kbNamespace 是用于从文件名派生确定性 UUID 的命名空间 UUID。
// 这个 UUID 无特殊含义，仅用于确保同一文件名生成相同的 UUIDv5。
var kbNamespace = uuid.MustParse("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

// docUUIDFromName 从文件名生成确定性 UUID（UUIDv5 / SHA-1）。
// 相同文件名总是产生相同 UUID，用于检测重复上传。
func docUUIDFromName(name string) string {
	return uuid.NewSHA1(kbNamespace, []byte(name)).String()
}

// FileInfo 记录知识库中上传文件的元数据。
type FileInfo struct {
	UUID       string    `json:"uuid"`
	Name       string    `json:"name"`        // 原始文件名
	DiskName   string    `json:"disk_name"`   // 磁盘上的文件名（uuid_原始名）
	Size       int64     `json:"size"`        // 文件大小（字节）
	Hash       string    `json:"hash"`        // 文件内容的 SHA256 哈希
	CreatedAt  time.Time `json:"created_at"`  // 上传时间
	ChunkCount int       `json:"chunk_count"` // 索引后的分块数
}

// Manager 管理本地知识库的全部生命周期。
// 线程安全，支持并发上传和删除操作。
type Manager struct {
	baseDir   string
	metaFile  string
	files     map[string]*FileInfo // uuid -> FileInfo
	mu        sync.RWMutex
	chunker   *rag.Chunker
	llmClient *llm.Client
	store     rag.Store
	graph     *rag.GraphStore
}

// NewManager 创建知识库管理器。
// baseDir: 知识库存放目录（自动创建）
// 启动时自动扫描并索引目录中的现有文件。
func NewManager(baseDir string, chunker *rag.Chunker, llmClient *llm.Client, store rag.Store, graph *rag.GraphStore) (*Manager, error) {
	// 确保目录存在
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("create knowledge base dir: %w", err)
	}
	metaFile := filepath.Join(baseDir, "kb_meta.json")
	m := &Manager{
		baseDir:   baseDir,
		metaFile:  metaFile,
		files:     make(map[string]*FileInfo),
		chunker:   chunker,
		llmClient: llmClient,
		store:     store,
		graph:     graph,
	}
	// 加载已有元数据
	if err := m.loadMeta(); err != nil {
		log.Printf("[KB] Warning: cannot load metadata: %v", err)
	}
	return m, nil
}

// LoadAndIndexExisting 扫描知识库目录，索引所有尚未索引的文件。
// 启动时调用，确保持久化的文件在 Agent 重启后仍可检索。
func (m *Manager) LoadAndIndexExisting(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	entries, err := os.ReadDir(m.baseDir)
	if err != nil {
		return fmt.Errorf("read kb dir: %w", err)
	}

	var indexed int
	for _, entry := range entries {
		if entry.IsDir() || entry.Name() == "kb_meta.json" {
			continue
		}
		// 检查是否已在元数据中
		found := false
		for _, fi := range m.files {
			if fi.DiskName == entry.Name() {
				found = true
				break
			}
		}
		if found {
			continue
		}

		// 新文件，自动编入索引
		fullPath := filepath.Join(m.baseDir, entry.Name())
		data, err := os.ReadFile(fullPath)
		if err != nil {
			log.Printf("[KB] Warning: cannot read %s: %v", entry.Name(), err)
			continue
		}

		// 推断原始文件名（磁盘名格式为 uuid_原始名）
		origName := entry.Name()
		if idx := strings.Index(entry.Name(), "_"); idx > 0 {
			if _, err := uuid.Parse(entry.Name()[:idx]); err == nil {
				origName = entry.Name()[idx+1:]
			}
		}

		hash := sha256.Sum256(data)
		fi := &FileInfo{
			UUID:      uuid.New().String(),
			Name:      origName,
			DiskName:  entry.Name(),
			Size:      int64(len(data)),
			Hash:      hex.EncodeToString(hash[:]),
			CreatedAt: time.Now(),
		}
		m.files[fi.UUID] = fi

		chunks := m.indexContent(ctx, string(data), fi)
		fi.ChunkCount = len(chunks)
		_ = chunks
		indexed++
	}

	if indexed > 0 {
		if err := m.saveMeta(); err != nil {
			log.Printf("[KB] Warning: save metadata: %v", err)
		}
		log.Printf("[KB] Indexed %d existing files on startup", indexed)
	} else {
		log.Printf("[KB] No new files to index on startup")
	}
	return nil
}

// Upload 上传并索引一个文件。
// 支持 .txt 和 .md 格式；其他格式会被拒绝。
// 返回文件 UUID 和索引后的分块数。
func (m *Manager) Upload(ctx context.Context, filename string, reader io.Reader) (*FileInfo, error) {
	// 验证文件格式
	ext := strings.ToLower(filepath.Ext(filename))
	if ext != ".txt" && ext != ".md" {
		return nil, fmt.Errorf("unsupported file type %q, only .txt and .md are supported", ext)
	}

	// 读取完整内容
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("read upload data: %w", err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("empty file")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// 从文件名生成确定性 UUID（相同文件名 -> 相同 UUID）
	fileUUID := docUUIDFromName(filename)
	hash := sha256.Sum256(data)
	hashStr := hex.EncodeToString(hash[:])

	// 检查是否已存在同名文档
	if existing, ok := m.files[fileUUID]; ok {
		if existing.Hash == hashStr {
			log.Printf("[KB] Skipping %q — unchanged (hash matches)", filename)
			cp := *existing
			return &cp, nil
		}
		log.Printf("[KB] Updating %q — content changed, re-indexing", filename)
		// 删除旧的向量 chunk
		if err := m.store.DeleteByDocUUID(ctx, fileUUID); err != nil {
			log.Printf("[KB] Warning: delete old chunks for %q: %v", filename, err)
		}
		// 删除旧的磁盘文件
		oldPath := filepath.Join(m.baseDir, existing.DiskName)
		if err := os.Remove(oldPath); err != nil && !os.IsNotExist(err) {
			log.Printf("[KB] Warning: remove old file for %q: %v", filename, err)
		}
		delete(m.files, fileUUID)
	}

	// 清理文件名（防止路径穿越）
	safeName := sanitizeFilename(filename)
	diskName := fmt.Sprintf("%s_%s", fileUUID, safeName)

	// 写入磁盘
	fullPath := filepath.Join(m.baseDir, diskName)
	if err := os.WriteFile(fullPath, data, 0644); err != nil {
		return nil, fmt.Errorf("write file: %w", err)
	}

	fi := &FileInfo{
		UUID:      fileUUID,
		Name:      filename,
		DiskName:  diskName,
		Size:      int64(len(data)),
		Hash:      hashStr,
		CreatedAt: time.Now(),
	}

	// 索引内容
	chunks := m.indexContent(ctx, string(data), fi)
	fi.ChunkCount = len(chunks)

	m.files[fileUUID] = fi
	if err := m.saveMeta(); err != nil {
		log.Printf("[KB] Warning: save metadata: %v", err)
	}

	log.Printf("[KB] Uploaded and indexed %q (%d chunks, %d bytes)", filename, len(chunks), len(data))
	return fi, nil
}

// ImportFile 从本地文件系统路径导入文件到知识库，相当于从磁盘直接复制+索引。
// filePath 可以是绝对路径或相对于工作目录的路径。
// 文件会被复制到知识库目录中，原始文件不受影响。
func (m *Manager) ImportFile(ctx context.Context, filePath string) (*FileInfo, error) {
	// 验证文件存在
	fi, err := os.Stat(filePath)
	if err != nil {
		return nil, fmt.Errorf("file not accessible: %w", err)
	}
	if fi.IsDir() {
		return nil, fmt.Errorf("path is a directory, not a file")
	}

	// 验证文件格式
	ext := strings.ToLower(filepath.Ext(fi.Name()))
	if ext != ".txt" && ext != ".md" {
		return nil, fmt.Errorf("unsupported file type %q, only .txt and .md are supported", ext)
	}

	// 读取文件内容
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("empty file")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	filename := filepath.Base(filePath)

	// 从文件名生成确定性 UUID
	fileUUID := docUUIDFromName(filename)
	hash := sha256.Sum256(data)
	hashStr := hex.EncodeToString(hash[:])

	// 检查是否已存在同名文档
	if existing, ok := m.files[fileUUID]; ok {
		if existing.Hash == hashStr {
			log.Printf("[KB] Skipping import of %q — unchanged (hash matches)", filename)
			cp := *existing
			return &cp, nil
		}
		log.Printf("[KB] Updating %q via import — content changed, re-indexing", filename)
		if err := m.store.DeleteByDocUUID(ctx, fileUUID); err != nil {
			log.Printf("[KB] Warning: delete old chunks for %q: %v", filename, err)
		}
		oldPath := filepath.Join(m.baseDir, existing.DiskName)
		if err := os.Remove(oldPath); err != nil && !os.IsNotExist(err) {
			log.Printf("[KB] Warning: remove old file for %q: %v", filename, err)
		}
		delete(m.files, fileUUID)
	}

	// 清理文件名并复制到知识库目录
	safeName := sanitizeFilename(filename)
	diskName := fmt.Sprintf("%s_%s", fileUUID, safeName)
	destPath := filepath.Join(m.baseDir, diskName)
	if err := os.WriteFile(destPath, data, 0644); err != nil {
		return nil, fmt.Errorf("copy file to kb dir: %w", err)
	}

	info := &FileInfo{
		UUID:      fileUUID,
		Name:      filename,
		DiskName:  diskName,
		Size:      int64(len(data)),
		Hash:      hashStr,
		CreatedAt: time.Now(),
	}

	// 索引内容（分块 + 嵌入 + 存储 + 图谱）
	chunks := m.indexContent(ctx, string(data), info)
	info.ChunkCount = len(chunks)

	m.files[fileUUID] = info
	if err := m.saveMeta(); err != nil {
		log.Printf("[KB] Warning: save metadata: %v", err)
	}

	log.Printf("[KB] Imported %q from local path (%d chunks, %d bytes)", info.Name, len(chunks), len(data))
	return info, nil
}

// ListFiles 返回所有已索引文件的元数据列表。
func (m *Manager) ListFiles() []*FileInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]*FileInfo, 0, len(m.files))
	for _, fi := range m.files {
		cp := *fi // 浅拷贝
		result = append(result, &cp)
	}
	// 按上传时间排序（最新的在前）
	sort.Slice(result, func(i, j int) bool {
		return result[i].CreatedAt.After(result[j].CreatedAt)
	})
	return result
}

// DeleteFile 删除指定 UUID 的文件，并从向量存储和知识图谱中移除。
func (m *Manager) DeleteFile(ctx context.Context, fileUUID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	fi, ok := m.files[fileUUID]
	if !ok {
		return fmt.Errorf("file %q not found", fileUUID)
	}

	// 从磁盘删除
	fullPath := filepath.Join(m.baseDir, fi.DiskName)
	if err := os.Remove(fullPath); err != nil && !os.IsNotExist(err) {
		log.Printf("[KB] Warning: remove file from disk: %v", err)
	}

	// 从向量存储删除
	if err := m.store.DeleteByDocUUID(ctx, fileUUID); err != nil {
		log.Printf("[KB] Warning: delete chunks from store: %v", err)
	}

	// 从元数据中移除
	delete(m.files, fileUUID)

	// 保存元数据
	if err := m.saveMeta(); err != nil {
		log.Printf("[KB] Warning: save metadata after delete: %v", err)
	}

	// 重建知识图谱
	m.rebuildGraph(ctx)

	log.Printf("[KB] Deleted file %q (%s)", fi.Name, fileUUID)
	return nil
}

// ReindexAll 清空所有索引并重新索引知识库目录中的所有文件。
// 当向量存储需要一致性修复时使用。
func (m *Manager) ReindexAll(ctx context.Context) (int, int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 删除所有现有文件的 chunks
	for uuid := range m.files {
		if err := m.store.DeleteByDocUUID(ctx, uuid); err != nil {
			log.Printf("[KB] Warning: delete chunks for %s: %v", uuid, err)
		}
	}

	// 读取目录中的所有文件
	entries, err := os.ReadDir(m.baseDir)
	if err != nil {
		return 0, 0, fmt.Errorf("read kb dir: %w", err)
	}

	// 清空已有文件元数据
	m.files = make(map[string]*FileInfo)
	m.graph.Clear()

	var totalChunks int
	var fileCount int

	for _, entry := range entries {
		if entry.IsDir() || entry.Name() == "kb_meta.json" {
			continue
		}

		fullPath := filepath.Join(m.baseDir, entry.Name())
		data, err := os.ReadFile(fullPath)
		if err != nil {
			log.Printf("[KB] Warning: cannot read %s: %v", entry.Name(), err)
			continue
		}

		// 推断原始文件名
		origName := entry.Name()
		if idx := strings.Index(entry.Name(), "_"); idx > 0 {
			if _, err := uuid.Parse(entry.Name()[:idx]); err == nil {
				origName = entry.Name()[idx+1:]
			}
		}

		hash := sha256.Sum256(data)
		fi := &FileInfo{
			UUID:      uuid.New().String(),
			Name:      origName,
			DiskName:  entry.Name(),
			Size:      int64(len(data)),
			Hash:      hex.EncodeToString(hash[:]),
			CreatedAt: time.Now(),
		}
		m.files[fi.UUID] = fi

		chunks := m.indexContent(ctx, string(data), fi)
		fi.ChunkCount = len(chunks)
		totalChunks += len(chunks)
		_ = chunks
		fileCount++
	}

	if err := m.saveMeta(); err != nil {
		log.Printf("[KB] Warning: save metadata after reindex: %v", err)
	}

	log.Printf("[KB] Reindexed %d files, %d chunks", fileCount, totalChunks)
	return fileCount, totalChunks, nil
}

// GetFile 按 UUID 获取文件信息。
func (m *Manager) GetFile(uuid string) *FileInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	fi, ok := m.files[uuid]
	if !ok {
		return nil
	}
	cp := *fi
	return &cp
}

// Count 返回已索引的文件数。
func (m *Manager) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.files)
}

// ==================== 内部方法 ====================

// indexContent 对文件内容执行分块、嵌入和存储。
// 遵循与 IndexSampleDocuments 相同的流程以保证一致性。
func (m *Manager) indexContent(ctx context.Context, content string, fi *FileInfo) []*types.Chunk {
	// 1. 分块
	chunks := m.chunker.Chunk(content, fi.Name)
	if len(chunks) == 0 {
		return nil
	}

	// 2. 设置元数据
	for i := range chunks {
		chunks[i].ID = fmt.Sprintf("%s-%d", fi.UUID, i)
		if chunks[i].Metadata == nil {
			chunks[i].Metadata = make(map[string]string)
		}
		chunks[i].Metadata["doc_uuid"] = fi.UUID
		chunks[i].Metadata["doc_hash"] = fi.Hash
	}

	// 3. 生成嵌入向量
	texts := make([]string, len(chunks))
	for i, ch := range chunks {
		texts[i] = ch.Content
	}
	if len(texts) > 0 {
		embeddings, err := m.llmClient.EmbedBatch(ctx, texts)
		if err != nil {
			log.Printf("[KB] Warning: embedding failed for %q: %v", fi.Name, err)
		} else {
			for i, emb := range embeddings {
				if emb != nil {
					chunks[i].Embedding = emb
				}
			}
		}
	}

	// 4. 存储到向量存储
	m.store.AddBatch(chunks)

	// 5. 更新知识图谱
	m.graph.BuildFromChunks(chunks, m.llmClient)

	return chunks
}

// rebuildGraph 从所有当前存储在 store 中的 chunks 重建知识图谱。
func (m *Manager) rebuildGraph(ctx context.Context) {
	m.graph.Clear()
	allChunks := m.store.GetAllChunks()
	if len(allChunks) > 0 {
		m.graph.BuildFromChunks(allChunks, m.llmClient)
	}
	// 无需保存 metadata，graph 在内存中
}

// loadMeta 从 JSON 文件加载文件元数据。
func (m *Manager) loadMeta() error {
	data, err := os.ReadFile(m.metaFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // 首次使用，无元数据正常
		}
		return err
	}
	var meta struct {
		Files []*FileInfo `json:"files"`
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		return fmt.Errorf("parse metadata: %w", err)
	}
	for _, fi := range meta.Files {
		m.files[fi.UUID] = fi
	}
	log.Printf("[KB] Loaded metadata for %d files", len(m.files))
	return nil
}

// saveMeta 将文件元数据持久化到 JSON 文件。
func (m *Manager) saveMeta() error {
	meta := struct {
		Files []*FileInfo `json:"files"`
	}{
		Files: make([]*FileInfo, 0, len(m.files)),
	}
	for _, fi := range m.files {
		meta.Files = append(meta.Files, fi)
	}
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(m.metaFile, data, 0644)
}

// sanitizeFilename 清理文件名，防止路径穿越和非法字符。
func sanitizeFilename(name string) string {
	// 替换路径分隔符
	name = strings.ReplaceAll(name, "/", "_")
	name = strings.ReplaceAll(name, "\\", "_")
	name = strings.ReplaceAll(name, "..", "_")
	// 移除可能导致问题的字符
	name = strings.Map(func(r rune) rune {
		if r >= 32 && r <= 126 && r != '<' && r != '>' && r != ':' && r != '"' && r != '|' && r != '?' && r != '*' {
			return r
		}
		return '_'
	}, name)
	// 限制长度
	if len(name) > 200 {
		name = name[:200]
	}
	return name
}
