# FinRAG

A Go-based Retrieval-Augmented Generation (RAG) system for financial documents
(10-Ks, earnings reports, balance sheets, invoices). Documents are extracted to
text, chunked with overlap, embedded via Voyage AI, stored in PostgreSQL with
pgvector (HNSW index), and retrieved by cosine similarity to ground Claude's
answers to financial questions.

> **Status:** Verified end-to-end — a real PDF upload flows through real Voyage
> embedding calls, real pgvector retrieval, and real Claude generation, all
> confirmed working together (not just unit-tested in isolation).

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
                 │  embedding   │  Voyage AI voyage-3-lite (512-dim)
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
# edit .env — set POSTGRES_PASSWORD, VOYAGE_API_KEY, ANTHROPIC_API_KEY

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

## Grounding in practice

FinRAG answers **only** from the chunks retrieved for a question, and says so
explicitly when the answer isn't in them — rather than filling the gap from
general knowledge. For a system meant to answer financial questions accurately,
this refusal-to-guess is the core reliability property: a fabricated figure is
worse than "not in the documents."

The following is a real tested exchange against a sample candidate profile
(`resume.pdf`) — a document that mentions "Kubernetes", "EKS", and "Helm"
repeatedly in its experience section, which makes it easy for a naive system to
infer a certification that was never claimed.

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What AWS or Kubernetes certifications does this person have?"}'
```

```json
{
  "answer": "Based on the provided document, this person holds one AWS certification — the AWS Certified Solutions Architect – Associate — which is listed as in progress rather than completed. The document does not contain any Kubernetes-specific certification. While it mentions hands-on work with Kubernetes, EKS, and Helm in the experience section, none of that indicates a formal Kubernetes certification, so I can't report one.",
  "sources": [
    {"filename": "resume.pdf", "content": "Certifications: AWS Certified Solutions Architect – Associate (in progress)...", "distance": 0.14}
  ]
}
```

Two things the model gets right by staying grounded in the retrieved context:

- It reports the AWS certification as **"in progress," not completed** — as the
  document actually states.
- It **explicitly states there is no Kubernetes certification**, instead of
  inferring one from the frequent "Kubernetes"/"EKS"/"Helm" mentions elsewhere
  in the document.

## Configuration

All configuration is via environment variables (no secrets in code):

| Variable            | Description                                  |
| ------------------- | -------------------------------------------- |
| `DATABASE_URL`      | PostgreSQL connection string                 |
| `VOYAGE_API_KEY`    | Voyage AI key for embeddings                 |
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
