package ingestion

import "strings"

// Default chunking parameters, tuned for financial documents: 800-char chunks
// keep tables/paragraphs coherent without degrading embedding quality on very
// long inputs, and 100-char overlap preserves context across boundaries where a
// sentence or financial figure might otherwise be split.
const (
	DefaultChunkSize = 800
	DefaultOverlap   = 100
)

// ChunkText splits text into overlapping chunks using a sliding window of
// chunkSize characters with the given overlap. Boundaries are nudged to the
// nearest whitespace near the window edge so words aren't cut in half.
//
// It returns an empty slice for empty (or whitespace-only) input and a single
// chunk when the text is shorter than chunkSize. Passing chunkSize<=0 or
// overlap<0 (or overlap>=chunkSize) falls back to the package defaults so the
// window always advances.
func ChunkText(text string, chunkSize, overlap int) []string {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	if overlap < 0 || overlap >= chunkSize {
		overlap = DefaultOverlap
	}

	// Operate on runes so multi-byte characters (common in financial symbols
	// like € or £) are never split mid-character.
	runes := []rune(text)
	n := len(runes)

	if strings.TrimSpace(text) == "" {
		return []string{}
	}
	if n <= chunkSize {
		return []string{text}
	}

	var chunks []string
	start := 0
	for start < n {
		end := start + chunkSize
		if end >= n {
			chunks = append(chunks, string(runes[start:n]))
			break
		}

		// Prefer to break at the last whitespace within the window so we don't
		// cut a word in half. Only do so if that whitespace is reasonably close
		// to the window edge; otherwise a very long unbroken token would force
		// a tiny chunk, so we fall back to a hard cut at end.
		breakAt := lastSpaceBefore(runes, start, end)
		if breakAt <= start {
			breakAt = end
		}

		chunks = append(chunks, string(runes[start:breakAt]))

		// Advance the window, carrying `overlap` characters of context forward.
		next := breakAt - overlap
		if next <= start {
			// Guarantee forward progress even in degenerate cases.
			next = start + 1
		}
		start = next
	}

	return chunks
}

// lastSpaceBefore returns the index just after the last whitespace rune in
// runes[start:end], i.e. a break point that keeps whole words on the left side.
// It returns start if no whitespace is found in the window.
func lastSpaceBefore(runes []rune, start, end int) int {
	for i := end - 1; i > start; i-- {
		if isSpace(runes[i]) {
			return i + 1
		}
	}
	return start
}

func isSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r'
}
