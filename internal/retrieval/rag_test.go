package retrieval

import (
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/ahmedfawzy21/finrag/internal/storage"
)

func chunk(filename, content string) storage.ChunkResult {
	return storage.ChunkResult{
		ChunkID:    uuid.New(),
		DocumentID: uuid.New(),
		Filename:   filename,
		Content:    content,
	}
}

func TestBuildUserPrompt(t *testing.T) {
	tests := []struct {
		name        string
		chunks      []storage.ChunkResult
		question    string
		wantContain []string
		wantAbsent  []string
	}{
		{
			name:        "empty chunks makes absence explicit",
			chunks:      nil,
			question:    "What was Q4 revenue?",
			wantContain: []string{"no relevant document excerpts were found", "Question: What was Q4 revenue?"},
			wantAbsent:  []string{"[Source 1"},
		},
		{
			name:        "single chunk includes source attribution",
			chunks:      []storage.ChunkResult{chunk("10k-2024.pdf", "Total revenue was $5.2B.")},
			question:    "What was revenue?",
			wantContain: []string{"[Source 1: 10k-2024.pdf]", "Total revenue was $5.2B.", "Question: What was revenue?"},
			wantAbsent:  []string{"[Source 2"},
		},
		{
			name: "multiple chunks are numbered and attributed in order",
			chunks: []storage.ChunkResult{
				chunk("balance.pdf", "Assets: 100"),
				chunk("earnings.docx", "Net income: 20"),
			},
			question:    "Summarize the financials.",
			wantContain: []string{"[Source 1: balance.pdf]", "Assets: 100", "[Source 2: earnings.docx]", "Net income: 20"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildUserPrompt(tt.chunks, tt.question)
			for _, want := range tt.wantContain {
				if !strings.Contains(got, want) {
					t.Errorf("prompt missing %q\n--- prompt ---\n%s", want, got)
				}
			}
			for _, absent := range tt.wantAbsent {
				if strings.Contains(got, absent) {
					t.Errorf("prompt unexpectedly contains %q\n--- prompt ---\n%s", absent, got)
				}
			}
		})
	}
}

// TestBuildUserPromptSourceOrder verifies sources are numbered to match the
// input order (Source 1 appears before Source 2 in the text).
func TestBuildUserPromptSourceOrder(t *testing.T) {
	chunks := []storage.ChunkResult{
		chunk("first.pdf", "alpha"),
		chunk("second.pdf", "beta"),
	}
	got := buildUserPrompt(chunks, "q")

	i1 := strings.Index(got, "[Source 1: first.pdf]")
	i2 := strings.Index(got, "[Source 2: second.pdf]")
	if i1 == -1 || i2 == -1 {
		t.Fatalf("expected both sources present, got:\n%s", got)
	}
	if i1 >= i2 {
		t.Errorf("Source 1 should appear before Source 2; got indexes %d and %d", i1, i2)
	}
}

// TestSystemPromptGrounding guards the two properties that matter most for
// financial accuracy: the model is told to answer only from context and to
// decline rather than guess.
func TestSystemPromptGrounding(t *testing.T) {
	lower := strings.ToLower(systemPrompt)
	if !strings.Contains(lower, "only") {
		t.Error("system prompt should restrict answers to the provided context")
	}
	if !strings.Contains(lower, "guess") && !strings.Contains(lower, "do not") {
		t.Error("system prompt should instruct the model not to guess when context is insufficient")
	}
}
