// Package api exposes the FinRAG HTTP surface: document ingestion, listing, and
// grounded querying, built on net/http's method+path routing (Go 1.22+).
package api

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ahmedfawzy21/finrag/internal/ingestion"
	"github.com/ahmedfawzy21/finrag/internal/retrieval"
	"github.com/ahmedfawzy21/finrag/internal/storage"
)

// maxUploadBytes caps an uploaded document at 32 MiB.
const maxUploadBytes = 32 << 20

// Server holds the dependencies the HTTP handlers need.
type Server struct {
	store    *storage.Store
	embedder *ingestion.Embedder
	rag      *retrieval.RAGEngine
	logger   *slog.Logger
}

// NewServer wires the handler dependencies together.
func NewServer(store *storage.Store, embedder *ingestion.Embedder, rag *retrieval.RAGEngine, logger *slog.Logger) *Server {
	return &Server{store: store, embedder: embedder, rag: rag, logger: logger}
}

// Handler builds the routed, middleware-wrapped http.Handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /documents", s.handleUpload)
	mux.HandleFunc("GET /documents", s.handleListDocuments)
	mux.HandleFunc("POST /query", s.handleQuery)
	return s.loggingMiddleware(mux)
}

// uploadResponse is returned after a successful ingestion.
type uploadResponse struct {
	ID         string `json:"id"`
	Filename   string `json:"filename"`
	SourceType string `json:"source_type"`
	Chunks     int    `json:"chunks"`
}

// handleUpload runs the extract → chunk → embed → store pipeline for a
// multipart-uploaded document and returns the new document ID.
func (s *Server) handleUpload(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid multipart upload: %v", err))
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		s.writeError(w, http.StatusBadRequest, "missing 'file' form field")
		return
	}
	defer file.Close()

	sourceType, err := sourceTypeFor(header.Filename)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Extractors operate on a file path, so persist the upload to a temp file
	// that preserves the extension for extension-based routing.
	tmpPath, cleanup, err := saveTemp(file, header.Filename)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to buffer upload: %v", err))
		return
	}
	defer cleanup()

	extractor, err := ingestion.ExtractorFor(tmpPath)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	text, err := extractor.Extract(ctx, tmpPath)
	if err != nil {
		s.writeError(w, http.StatusUnprocessableEntity, fmt.Sprintf("failed to extract text: %v", err))
		return
	}

	chunks := ingestion.ChunkText(text, ingestion.DefaultChunkSize, ingestion.DefaultOverlap)
	if len(chunks) == 0 {
		s.writeError(w, http.StatusUnprocessableEntity, "document produced no text to index")
		return
	}

	embeddings, err := s.embedder.EmbedBatch(ctx, chunks, ingestion.InputTypeDocument)
	if err != nil {
		s.writeError(w, http.StatusBadGateway, fmt.Sprintf("failed to embed chunks: %v", err))
		return
	}

	docID, err := s.store.InsertDocument(ctx, header.Filename, sourceType, nil)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to store document: %v", err))
		return
	}

	for i, content := range chunks {
		if err := s.store.InsertChunk(ctx, docID, i, content, embeddings[i]); err != nil {
			s.writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to store chunk %d: %v", i, err))
			return
		}
	}

	s.writeJSON(w, http.StatusCreated, uploadResponse{
		ID:         docID.String(),
		Filename:   header.Filename,
		SourceType: sourceType,
		Chunks:     len(chunks),
	})
}

// handleListDocuments returns a summary of every ingested document.
func (s *Server) handleListDocuments(w http.ResponseWriter, r *http.Request) {
	docs, err := s.store.ListDocuments(r.Context())
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to list documents: %v", err))
		return
	}
	s.writeJSON(w, http.StatusOK, map[string]any{"documents": docs})
}

// queryRequest is the JSON body for POST /query.
type queryRequest struct {
	Question string `json:"question"`
	TopK     int    `json:"top_k,omitempty"`
}

// sourceChunk is the JSON-facing view of a retrieved chunk.
type sourceChunk struct {
	ChunkID    string  `json:"chunk_id"`
	DocumentID string  `json:"document_id"`
	Filename   string  `json:"filename"`
	Content    string  `json:"content"`
	Distance   float64 `json:"distance"`
}

// queryResponse is the JSON body returned by POST /query.
type queryResponse struct {
	Answer  string        `json:"answer"`
	Sources []sourceChunk `json:"sources"`
}

// handleQuery answers a question grounded in the retrieved document chunks.
func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	var req queryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON body: %v", err))
		return
	}
	if strings.TrimSpace(req.Question) == "" {
		s.writeError(w, http.StatusBadRequest, "'question' must not be empty")
		return
	}

	answer, err := s.rag.Query(r.Context(), req.Question, req.TopK)
	if err != nil {
		s.writeError(w, http.StatusBadGateway, fmt.Sprintf("failed to answer query: %v", err))
		return
	}

	sources := make([]sourceChunk, 0, len(answer.SourceChunks))
	for _, c := range answer.SourceChunks {
		sources = append(sources, sourceChunk{
			ChunkID:    c.ChunkID.String(),
			DocumentID: c.DocumentID.String(),
			Filename:   c.Filename,
			Content:    c.Content,
			Distance:   c.Distance,
		})
	}

	s.writeJSON(w, http.StatusOK, queryResponse{Answer: answer.Text, Sources: sources})
}

// sourceTypeFor maps a filename to its stored source_type.
func sourceTypeFor(filename string) (string, error) {
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".pdf":
		return "pdf", nil
	case ".docx":
		return "docx", nil
	case ".png", ".jpg", ".jpeg":
		return "image", nil
	default:
		return "", fmt.Errorf("unsupported file type %q: supported extensions are .pdf, .docx, .png, .jpg, .jpeg", filepath.Ext(filename))
	}
}

// saveTemp writes an uploaded file to a temp file whose name preserves the
// original extension, and returns the path plus a cleanup func.
func saveTemp(src io.Reader, filename string) (string, func(), error) {
	ext := filepath.Ext(filename)
	f, err := os.CreateTemp("", "finrag-upload-*"+ext)
	if err != nil {
		return "", nil, err
	}
	cleanup := func() {
		f.Close()
		os.Remove(f.Name())
	}
	if _, err := io.Copy(f, src); err != nil {
		cleanup()
		return "", nil, err
	}
	if err := f.Close(); err != nil {
		cleanup()
		return "", nil, err
	}
	return f.Name(), cleanup, nil
}

// writeJSON writes v as a JSON response with the given status.
func (s *Server) writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		s.logger.Error("failed to encode response", "error", err)
	}
}

// writeError writes a JSON error body with the given status.
func (s *Server) writeError(w http.ResponseWriter, status int, message string) {
	s.writeJSON(w, status, map[string]string{"error": message})
}

// statusRecorder captures the response status for logging.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// loggingMiddleware logs each request's method, path, status, and duration via
// structured logging.
func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		s.logger.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rec.status,
			"duration", time.Since(start).String(),
			"remote", r.RemoteAddr,
		)
	})
}
