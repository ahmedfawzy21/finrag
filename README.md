# FinRAG

A Go-based Retrieval-Augmented Generation (RAG) system for financial documents
(10-Ks, earnings reports, balance sheets, invoices). Documents are extracted to
text, chunked with overlap, embedded via OpenAI, stored in PostgreSQL with
pgvector (HNSW index), and retrieved by cosine similarity to ground Claude's
answers to financial questions.

## Architecture

```
                 ┌──────────────┐
 upload  ──────► │  extraction  │  PDF → pdftotext, DOCX → zip+XML,
 (PDF/DOCX/img)  │  (ingestion) │  image → Claude vision
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │   chunking   │  800-char sliding window, 100-char overlap
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  embedding   │  OpenAI text-embedding-3-small (1536-dim)
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │   storage    │  PostgreSQL + pgvector, HNSW cosine index
                 └──────┬───────┘
                        ▼
 question ─────► ┌──────────────┐
                 │  retrieval   │  embed question → top-K cosine search
                 │    + RAG     │  → grounded prompt → Claude (claude-opus-4-8)
                 └──────────────┘
```

Package layout (standard Go project layout):

- `cmd/server` — HTTP entrypoint, config wiring, graceful shutdown.
- `internal/storage` — PostgreSQL + pgvector persistence.
- `internal/ingestion` — extraction, chunking, embedding.
- `internal/retrieval` — the RAG engine (retrieval + generation).
- `internal/api` — REST handlers and request logging.

## Key design decisions

**1. HNSW over IVFFlat for the vector index.** HNSW gives a better
recall/speed tradeoff at this dataset size and requires no training step. IVFFlat
needs representative data available up front to build its cluster lists, which is
awkward for a system that ingests documents incrementally; HNSW builds
incrementally as rows are inserted.

**2. pgvector over a dedicated vector DB (e.g. Pinecone).** A single PostgreSQL
database holds both the relational metadata (documents, chunk indexes,
timestamps) and the vectors, so there is no extra infrastructure to run or
synchronize. Performance is more than adequate at this scale, and it avoids
vendor lock-in.

**3. 800-char chunks with 100-char overlap for financial documents.** 800
characters balance context completeness (keeping a table row or paragraph
together) against embedding quality, which degrades on very long chunks. The
100-char overlap prevents losing context at chunk boundaries where a sentence or
a financial figure might otherwise be split across two chunks.

## Quickstart

Requires Docker + Docker Compose.

```bash
# 1. Configure secrets.
cp .env.example .env
# edit .env — set POSTGRES_PASSWORD, OPENAI_API_KEY, ANTHROPIC_API_KEY

# 2. Start Postgres (with pgvector) and the app.
docker compose up --build
```

The API listens on `http://localhost:8080`.

### Upload a document

```bash
curl -X POST http://localhost:8080/documents \
  -F "file=@/path/to/annual-report.pdf"
# → {"id":"...","filename":"annual-report.pdf","source_type":"pdf","chunks":42}
```

Supported types: `.pdf`, `.docx`, `.png`, `.jpg`, `.jpeg`.

### List documents

```bash
curl http://localhost:8080/documents
# → {"documents":[{"ID":"...","Filename":"annual-report.pdf","SourceType":"pdf",...}]}
```

### Ask a question

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What was total revenue in fiscal 2024?","top_k":5}'
# → {"answer":"...","sources":[{"filename":"annual-report.pdf","content":"...","distance":0.12}]}
```

`top_k` is optional (default 5).

## Configuration

All configuration is via environment variables (no secrets in code):

| Variable            | Description                                  |
| ------------------- | -------------------------------------------- |
| `DATABASE_URL`      | PostgreSQL connection string                 |
| `OPENAI_API_KEY`    | OpenAI key for embeddings                    |
| `ANTHROPIC_API_KEY` | Anthropic key for vision + generation        |
| `PORT`              | HTTP port (default `8080`)                   |

## Development

```bash
go build ./...
go vet ./...
go test ./...
```

Local PDF extraction requires `pdftotext` (poppler-utils):

```bash
apt install poppler-utils   # Debian/Ubuntu
```
