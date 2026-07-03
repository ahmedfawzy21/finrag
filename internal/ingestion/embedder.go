package ingestion

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// EmbeddingModel is voyage-3-lite, whose 512-dim output matches the
// vector(512) column in the storage schema.
const EmbeddingModel = "voyage-3-lite"

// voyageEndpoint is the Voyage AI embeddings REST endpoint. Voyage ships no
// official Go SDK, so we call it directly over net/http.
const voyageEndpoint = "https://api.voyageai.com/v1/embeddings"

// Voyage input_type values. Voyage recommends distinguishing stored documents
// from search queries for better retrieval accuracy.
const (
	InputTypeDocument = "document"
	InputTypeQuery    = "query"
)

// Embedder calls the Voyage AI REST API to produce embedding vectors for text.
type Embedder struct {
	apiKey     string
	httpClient *http.Client
	model      string
	// endpoint is the URL EmbedBatch POSTs to. It defaults to voyageEndpoint
	// and is overridable so tests can point at a local httptest server.
	endpoint string
}

// NewEmbedder returns an Embedder using the given Voyage AI API key.
func NewEmbedder(apiKey string) *Embedder {
	return &Embedder{
		apiKey:     apiKey,
		httpClient: &http.Client{Timeout: 30 * time.Second},
		model:      EmbeddingModel,
		endpoint:   voyageEndpoint,
	}
}

// embeddingRequest is the JSON body sent to the Voyage embeddings endpoint.
type embeddingRequest struct {
	Input     []string `json:"input"`
	Model     string   `json:"model"`
	InputType string   `json:"input_type"`
}

// embeddingResponse is the JSON body returned by the Voyage embeddings endpoint.
type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// voyageErrorResponse captures Voyage's error message format for non-200
// responses so we can surface a meaningful error rather than a bare status code.
type voyageErrorResponse struct {
	Detail  string `json:"detail"`
	Error   string `json:"error"`
	Message string `json:"message"`
}

// buildRequestBody constructs the Voyage request JSON for the given texts and
// input type. It is a pure function so request construction can be unit-tested
// without any HTTP call.
func buildRequestBody(texts []string, model, inputType string) ([]byte, error) {
	return json.Marshal(embeddingRequest{
		Input:     texts,
		Model:     model,
		InputType: inputType,
	})
}

// parseResponseBody parses a Voyage embeddings response and returns the
// embeddings ordered to match the input order. Voyage tags each embedding with
// the index of its input text; we place by that index rather than trusting the
// response order.
func parseResponseBody(body []byte, expected int) ([][]float32, error) {
	var resp embeddingResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse embedding response: %w", err)
	}
	if len(resp.Data) != expected {
		return nil, fmt.Errorf("failed to parse embedding response: expected %d vectors, got %d", expected, len(resp.Data))
	}

	out := make([][]float32, expected)
	for _, emb := range resp.Data {
		if emb.Index < 0 || emb.Index >= expected {
			return nil, fmt.Errorf("failed to parse embedding response: out-of-range index %d", emb.Index)
		}
		if out[emb.Index] != nil {
			return nil, fmt.Errorf("failed to parse embedding response: duplicate index %d", emb.Index)
		}
		out[emb.Index] = emb.Embedding
	}
	return out, nil
}

// Embed returns the embedding vector for a single piece of text. inputType
// should be InputTypeDocument when embedding chunks for storage and
// InputTypeQuery when embedding a user's question for search.
func (e *Embedder) Embed(ctx context.Context, text string, inputType string) ([]float32, error) {
	vectors, err := e.EmbedBatch(ctx, []string{text}, inputType)
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// EmbedBatch embeds multiple texts in a single API call, reducing latency and
// cost during bulk ingestion. The returned slice is aligned to the input order.
// inputType should be InputTypeDocument for stored chunks and InputTypeQuery
// for search queries.
func (e *Embedder) EmbedBatch(ctx context.Context, texts []string, inputType string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	body, err := buildRequestBody(texts, e.model, inputType)
	if err != nil {
		return nil, fmt.Errorf("failed to build embedding request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to build embedding request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call Voyage embeddings API: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read Voyage response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("voyage embeddings request failed: %s", voyageError(resp.StatusCode, respBody))
	}

	return parseResponseBody(respBody, len(texts))
}

// voyageError renders a human-readable error for a non-200 Voyage response,
// extracting Voyage's error message from the body when present and calling out
// rate limiting explicitly.
func voyageError(status int, body []byte) string {
	msg := extractVoyageErrorMessage(body)

	if status == http.StatusTooManyRequests {
		detail := "rate limited (HTTP 429) — this may be the 200M free token tier limit; wait and retry"
		if msg != "" {
			return detail + ": " + msg
		}
		return detail
	}

	if msg != "" {
		return fmt.Sprintf("HTTP %d: %s", status, msg)
	}
	if len(body) > 0 {
		return fmt.Sprintf("HTTP %d: %s", status, string(body))
	}
	return fmt.Sprintf("HTTP %d", status)
}

// extractVoyageErrorMessage pulls the message out of Voyage's error JSON,
// tolerating the few shapes the API uses. It returns "" when the body is not
// recognizable error JSON.
func extractVoyageErrorMessage(body []byte) string {
	var e voyageErrorResponse
	if err := json.Unmarshal(body, &e); err != nil {
		return ""
	}
	switch {
	case e.Detail != "":
		return e.Detail
	case e.Error != "":
		return e.Error
	case e.Message != "":
		return e.Message
	default:
		return ""
	}
}
