package storage

import (
	"strings"
	"testing"
)

// TestSearchSimilarChunksQuery verifies the pure query-construction helper.
// Full integration testing against a real database happens in Stage 5.
func TestSearchSimilarChunksQuery(t *testing.T) {
	query := searchSimilarChunksQuery()

	tests := []struct {
		name     string
		contains string
	}{
		{"cosine distance operator", "<=>"},
		{"query embedding parameter", "$1"},
		{"limit parameter", "$2"},
		{"ascending order for nearest-first", "ORDER BY"},
		{"joins documents for filename", "JOIN documents"},
		{"selects content", "c.content"},
		{"applies a limit", "LIMIT"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !strings.Contains(query, tt.contains) {
				t.Errorf("expected search query to contain %q, got:\n%s", tt.contains, query)
			}
		})
	}
}
