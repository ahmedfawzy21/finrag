package ingestion

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestBuildRequestBody(t *testing.T) {
	body, err := buildRequestBody([]string{"one", "two"}, "voyage-3-lite", InputTypeDocument)
	if err != nil {
		t.Fatalf("buildRequestBody returned error: %v", err)
	}

	var got embeddingRequest
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("could not unmarshal built body: %v", err)
	}

	want := embeddingRequest{
		Input:     []string{"one", "two"},
		Model:     "voyage-3-lite",
		InputType: "document",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("request body = %+v, want %+v", got, want)
	}

	// Verify the JSON field names match Voyage's expected keys.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("could not unmarshal built body to map: %v", err)
	}
	for _, key := range []string{"input", "model", "input_type"} {
		if _, ok := raw[key]; !ok {
			t.Errorf("request body missing key %q", key)
		}
	}
}

func TestParseResponseBodyReordersByIndex(t *testing.T) {
	// Embeddings arrive out of order (index 2, 0, 1); the parser must restore
	// input order using the index field.
	body := []byte(`{
		"data": [
			{"embedding": [2.0, 2.1], "index": 2},
			{"embedding": [0.0, 0.1], "index": 0},
			{"embedding": [1.0, 1.1], "index": 1}
		],
		"model": "voyage-3-lite",
		"usage": {"total_tokens": 9}
	}`)

	got, err := parseResponseBody(body, 3)
	if err != nil {
		t.Fatalf("parseResponseBody returned error: %v", err)
	}

	want := [][]float32{
		{0.0, 0.1},
		{1.0, 1.1},
		{2.0, 2.1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("reordered embeddings = %v, want %v", got, want)
	}
}

func TestParseResponseBodyCountMismatch(t *testing.T) {
	body := []byte(`{"data": [{"embedding": [1.0], "index": 0}], "model": "voyage-3-lite"}`)
	if _, err := parseResponseBody(body, 2); err == nil {
		t.Error("expected error on count mismatch, got nil")
	}
}

func TestParseResponseBodyOutOfRangeIndex(t *testing.T) {
	body := []byte(`{"data": [{"embedding": [1.0], "index": 5}], "model": "voyage-3-lite"}`)
	if _, err := parseResponseBody(body, 1); err == nil {
		t.Error("expected error on out-of-range index, got nil")
	}
}

func TestEmbedBatchAgainstMockServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization header = %q, want %q", got, "Bearer test-key")
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type header = %q, want application/json", got)
		}

		var req embeddingRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("could not decode request: %v", err)
		}
		if req.InputType != InputTypeQuery {
			t.Errorf("input_type = %q, want %q", req.InputType, InputTypeQuery)
		}
		if len(req.Input) != 2 {
			t.Errorf("input len = %d, want 2", len(req.Input))
		}

		// Respond with scrambled index order to exercise reordering end to end.
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"data": [
				{"embedding": [9.0], "index": 1},
				{"embedding": [8.0], "index": 0}
			],
			"model": "voyage-3-lite",
			"usage": {"total_tokens": 4}
		}`))
	}))
	defer srv.Close()

	// Point the embedder at the test server instead of the real Voyage endpoint.
	e := &Embedder{
		apiKey:     "test-key",
		httpClient: srv.Client(),
		model:      "voyage-3-lite",
		endpoint:   srv.URL,
	}

	got, err := e.EmbedBatch(context.Background(), []string{"alpha", "beta"}, InputTypeQuery)
	if err != nil {
		t.Fatalf("EmbedBatch returned error: %v", err)
	}

	want := [][]float32{{8.0}, {9.0}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("embeddings = %v, want %v", got, want)
	}
}

func TestEmbedBatchNonOKStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"detail": "rate limit exceeded"}`))
	}))
	defer srv.Close()

	e := &Embedder{apiKey: "test-key", httpClient: srv.Client(), model: "voyage-3-lite", endpoint: srv.URL}
	_, err := e.EmbedBatch(context.Background(), []string{"x"}, InputTypeDocument)
	if err == nil {
		t.Fatal("expected error on 429 status, got nil")
	}
}

func TestEmbedBatchEmptyInput(t *testing.T) {
	e := NewEmbedder("test-key")
	got, err := e.EmbedBatch(context.Background(), nil, InputTypeDocument)
	if err != nil {
		t.Fatalf("EmbedBatch on empty input returned error: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("EmbedBatch on empty input = %v, want empty", got)
	}
}
