// Package storage implements the PostgreSQL + pgvector persistence layer for
// FinRAG: documents, their text chunks, and the chunk embeddings used for
// cosine-similarity retrieval.
package storage

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

//go:embed schema.sql
var schemaSQL string

// Store wraps a pgx connection pool and exposes the persistence operations
// FinRAG needs.
type Store struct {
	pool *pgxpool.Pool
}

// ChunkResult is a single row returned from a similarity search: the chunk
// itself plus the owning document's filename and the cosine distance to the
// query embedding (smaller is closer).
type ChunkResult struct {
	ChunkID    uuid.UUID
	DocumentID uuid.UUID
	Filename   string
	Content    string
	Distance   float64
}

// DocumentSummary is a lightweight view of an ingested document for listings.
type DocumentSummary struct {
	ID         uuid.UUID
	Filename   string
	SourceType string
	UploadedAt string
	ChunkCount int
}

// NewStore connects to PostgreSQL at connString, verifies connectivity with a
// ping, and applies the embedded schema migration (extension, tables, HNSW
// index). It is safe to call repeatedly since the migration is idempotent.
func NewStore(ctx context.Context, connString string) (*Store, error) {
	pool, err := pgxpool.New(ctx, connString)
	if err != nil {
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	if _, err := pool.Exec(ctx, schemaSQL); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to apply schema migration: %w", err)
	}

	return &Store{pool: pool}, nil
}

// Close releases the underlying connection pool.
func (s *Store) Close() {
	s.pool.Close()
}

// InsertDocument inserts a document row and returns its generated UUID.
func (s *Store) InsertDocument(ctx context.Context, filename, sourceType string, metadata map[string]any) (uuid.UUID, error) {
	var metadataJSON []byte
	if metadata != nil {
		var err error
		metadataJSON, err = json.Marshal(metadata)
		if err != nil {
			return uuid.Nil, fmt.Errorf("failed to marshal document metadata: %w", err)
		}
	}

	var id uuid.UUID
	err := s.pool.QueryRow(ctx,
		`INSERT INTO documents (filename, source_type, metadata)
		 VALUES ($1, $2, $3)
		 RETURNING id`,
		filename, sourceType, metadataJSON,
	).Scan(&id)
	if err != nil {
		return uuid.Nil, fmt.Errorf("failed to insert document: %w", err)
	}

	return id, nil
}

// InsertChunk inserts a single chunk with its embedding for a document.
func (s *Store) InsertChunk(ctx context.Context, docID uuid.UUID, chunkIndex int, content string, embedding []float32) error {
	_, err := s.pool.Exec(ctx,
		`INSERT INTO chunks (document_id, chunk_index, content, embedding)
		 VALUES ($1, $2, $3, $4)`,
		docID, chunkIndex, content, pgvector.NewVector(embedding),
	)
	if err != nil {
		return fmt.Errorf("failed to insert chunk %d for document %s: %w", chunkIndex, docID, err)
	}
	return nil
}

// searchSimilarChunksQuery builds the cosine-similarity search SQL. It is a
// pure function so the query construction can be unit-tested without a live DB.
// The query embedding binds to $1 and the row limit to $2. pgvector's `<=>`
// operator computes cosine distance; ordering ascending yields nearest first.
func searchSimilarChunksQuery() string {
	return `SELECT c.id, c.document_id, d.filename, c.content,
	               c.embedding <=> $1 AS distance
	        FROM chunks c
	        JOIN documents d ON d.id = c.document_id
	        ORDER BY c.embedding <=> $1 ASC
	        LIMIT $2`
}

// SearchSimilarChunks returns the topmost `limit` chunks nearest to
// queryEmbedding by cosine distance.
func (s *Store) SearchSimilarChunks(ctx context.Context, queryEmbedding []float32, limit int) ([]ChunkResult, error) {
	rows, err := s.pool.Query(ctx, searchSimilarChunksQuery(),
		pgvector.NewVector(queryEmbedding), limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query similar chunks: %w", err)
	}
	defer rows.Close()

	var results []ChunkResult
	for rows.Next() {
		var r ChunkResult
		if err := rows.Scan(&r.ChunkID, &r.DocumentID, &r.Filename, &r.Content, &r.Distance); err != nil {
			return nil, fmt.Errorf("failed to scan chunk result: %w", err)
		}
		results = append(results, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("failed to iterate chunk results: %w", err)
	}

	return results, nil
}

// ListDocuments returns a summary of every ingested document with its chunk
// count, newest first.
func (s *Store) ListDocuments(ctx context.Context) ([]DocumentSummary, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT d.id, d.filename, d.source_type, d.uploaded_at,
		        COUNT(c.id) AS chunk_count
		 FROM documents d
		 LEFT JOIN chunks c ON c.document_id = d.id
		 GROUP BY d.id
		 ORDER BY d.uploaded_at DESC`)
	if err != nil {
		return nil, fmt.Errorf("failed to query documents: %w", err)
	}
	defer rows.Close()

	var docs []DocumentSummary
	for rows.Next() {
		var d DocumentSummary
		var uploadedAt time.Time
		if err := rows.Scan(&d.ID, &d.Filename, &d.SourceType, &uploadedAt, &d.ChunkCount); err != nil {
			return nil, fmt.Errorf("failed to scan document summary: %w", err)
		}
		d.UploadedAt = uploadedAt.Format(time.RFC3339)
		docs = append(docs, d)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("failed to iterate documents: %w", err)
	}

	return docs, nil
}
