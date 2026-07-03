// Package retrieval implements the RAG (retrieval-augmented generation) engine:
// it embeds a question, retrieves the nearest stored chunks by cosine
// similarity, and grounds Claude's answer in those chunks.
package retrieval

import (
	"context"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/ahmedfawzy21/finrag/internal/ingestion"
	"github.com/ahmedfawzy21/finrag/internal/storage"
)

// DefaultTopK is the number of chunks retrieved when the caller does not
// specify a positive topK.
const DefaultTopK = 5

// systemPrompt constrains Claude to answer only from the retrieved context.
// Being explicit that it must decline rather than guess matters for financial
// accuracy — a fabricated figure is worse than "not found in the documents".
const systemPrompt = `You are a financial document assistant. You answer questions strictly from the ` +
	`provided context, which consists of excerpts retrieved from the user's financial documents ` +
	`(10-Ks, earnings reports, balance sheets, invoices).

Rules:
- Answer ONLY using information present in the provided context.
- If the context does not contain the information needed to answer, say so plainly ` +
	`(e.g. "The provided documents do not contain that information.") — do NOT guess or draw on ` +
	`general knowledge.
- Preserve numbers, dates, and figures exactly as they appear in the context.
- When relevant, cite the source filename you drew the answer from.`

// Answer is the result of a RAG query: the generated text plus the chunks that
// were retrieved and used as grounding context.
type Answer struct {
	Text         string
	SourceChunks []storage.ChunkResult
}

// RAGEngine ties together storage (vector search), the embedder (to embed the
// question with the same model used for ingestion), and Claude (for grounded
// generation).
type RAGEngine struct {
	store    *storage.Store
	embedder *ingestion.Embedder
	client   anthropic.Client
	model    anthropic.Model
}

// NewRAGEngine constructs a RAGEngine. It reuses the caller-provided store and
// embedder and creates an Anthropic client from the given API key.
func NewRAGEngine(store *storage.Store, embedder *ingestion.Embedder, anthropicAPIKey string) *RAGEngine {
	return &RAGEngine{
		store:    store,
		embedder: embedder,
		client:   anthropic.NewClient(option.WithAPIKey(anthropicAPIKey)),
		model:    anthropic.ModelClaudeOpus4_8,
	}
}

// Query embeds the question, retrieves the topK nearest chunks, and asks Claude
// to answer grounded in those chunks.
func (r *RAGEngine) Query(ctx context.Context, question string, topK int) (*Answer, error) {
	if topK <= 0 {
		topK = DefaultTopK
	}

	queryEmbedding, err := r.embedder.Embed(ctx, question, ingestion.InputTypeQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to embed question: %w", err)
	}

	chunks, err := r.store.SearchSimilarChunks(ctx, queryEmbedding, topK)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve similar chunks: %w", err)
	}

	userPrompt := buildUserPrompt(chunks, question)

	msg, err := r.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     r.model,
		MaxTokens: 2048,
		System:    []anthropic.TextBlockParam{{Text: systemPrompt}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(userPrompt)),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to generate answer: %w", err)
	}

	var sb strings.Builder
	for _, block := range msg.Content {
		if text, ok := block.AsAny().(anthropic.TextBlock); ok {
			sb.WriteString(text.Text)
		}
	}

	return &Answer{
		Text:         sb.String(),
		SourceChunks: chunks,
	}, nil
}

// buildUserPrompt formats the retrieved chunks as an attributed context block
// followed by the question. It is a pure function so prompt construction can be
// unit-tested without any API call. When no chunks were retrieved, it makes the
// absence explicit so the model follows the "say you don't know" rule.
func buildUserPrompt(chunks []storage.ChunkResult, question string) string {
	var sb strings.Builder

	sb.WriteString("Context from the financial documents:\n\n")
	if len(chunks) == 0 {
		sb.WriteString("(no relevant document excerpts were found)\n")
	} else {
		for i, c := range chunks {
			fmt.Fprintf(&sb, "[Source %d: %s]\n%s\n\n", i+1, c.Filename, c.Content)
		}
	}

	sb.WriteString("Question: ")
	sb.WriteString(question)

	return sb.String()
}
