package ingestion

import (
	"fmt"
	"strings"
	"testing"
)

func TestChunkText(t *testing.T) {
	tests := []struct {
		name      string
		text      string
		chunkSize int
		overlap   int
		want      []string
	}{
		{
			name:      "empty string yields empty slice",
			text:      "",
			chunkSize: 10,
			overlap:   2,
			want:      []string{},
		},
		{
			name:      "whitespace-only yields empty slice",
			text:      "   \n\t ",
			chunkSize: 10,
			overlap:   2,
			want:      []string{},
		},
		{
			name:      "text shorter than chunk size is one chunk",
			text:      "short text",
			chunkSize: 100,
			overlap:   10,
			want:      []string{"short text"},
		},
		{
			name:      "text exactly chunk size is one chunk",
			text:      "abcdefghij", // 10 chars
			chunkSize: 10,
			overlap:   3,
			want:      []string{"abcdefghij"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ChunkText(tt.text, tt.chunkSize, tt.overlap)
			if len(got) != len(tt.want) {
				t.Fatalf("ChunkText() = %#v (len %d), want %#v (len %d)", got, len(got), tt.want, len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("chunk %d = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

// TestChunkTextOverlap verifies that consecutive chunks share exactly `overlap`
// characters: the tail of chunk N is the head of chunk N+1.
func TestChunkTextOverlap(t *testing.T) {
	// 200 space-separated 8-char words.
	var b strings.Builder
	for i := 0; i < 200; i++ {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "word%04d", i)
	}
	text := b.String()

	chunkSize, overlap := 100, 20
	chunks := ChunkText(text, chunkSize, overlap)
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}

	for i := 0; i < len(chunks)-1; i++ {
		cur := []rune(chunks[i])
		next := []rune(chunks[i+1])
		if len(cur) < overlap || len(next) < overlap {
			continue
		}
		tail := string(cur[len(cur)-overlap:])
		head := string(next[:overlap])
		if tail != head {
			t.Errorf("chunk %d tail %q does not match chunk %d head %q", i, tail, i+1, head)
		}
	}
}

// TestChunkTextWordBoundary verifies no chunk ends mid-word when a space
// boundary was available. (A chunk may *start* mid-word: the fixed-size overlap
// carries a fixed character count back, which is expected and acceptable — the
// spec's requirement is about not ending mid-word.)
func TestChunkTextWordBoundary(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 200; i++ {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "word%04d", i)
	}
	text := b.String()

	chunks := ChunkText(text, 100, 20)
	for ci := 0; ci < len(chunks)-1; ci++ { // every chunk except the last
		chunk := chunks[ci]
		// Breaking at a space means the chunk ends with whitespace...
		if !strings.HasSuffix(chunk, " ") {
			t.Errorf("chunk %d does not end at a word boundary: %q", ci, chunk)
		}
		// ...and its last token is therefore a complete 8-char word.
		fields := strings.Fields(chunk)
		last := fields[len(fields)-1]
		if len(last) != len("word0000") {
			t.Errorf("chunk %d ends with partial word %q", ci, last)
		}
	}
}

// TestChunkTextLongUnbrokenToken verifies forward progress (no infinite loop and
// full coverage) when a single token is longer than the window, so no
// whitespace break point exists.
func TestChunkTextLongUnbrokenToken(t *testing.T) {
	text := strings.Repeat("x", 2500) // no spaces at all
	chunks := ChunkText(text, 800, 100)
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks for long unbroken token, got %d", len(chunks))
	}
	// Reassembling by stripping the repeated overlap must reproduce the input.
	if strings.Count(strings.Join(chunks, ""), "x") < len(text) {
		t.Errorf("chunks do not cover the full input")
	}
}
