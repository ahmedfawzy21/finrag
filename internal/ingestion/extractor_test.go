package ingestion

import (
	"strings"
	"testing"
)

// TestClassifyExtension covers ExtractorFor's file-extension routing logic as a
// pure function, without constructing any extractor (which would require
// pdftotext or an API key).
func TestClassifyExtension(t *testing.T) {
	tests := []struct {
		name     string
		filePath string
		want     extractorKind
	}{
		{"pdf lowercase", "report.pdf", kindPDF},
		{"pdf uppercase", "REPORT.PDF", kindPDF},
		{"docx", "filing.docx", kindDOCX},
		{"png", "scan.png", kindImage},
		{"jpg", "scan.jpg", kindImage},
		{"jpeg", "scan.jpeg", kindImage},
		{"jpeg mixed case", "Scan.JPEG", kindImage},
		{"path with dirs", "/data/uploads/10-K.pdf", kindPDF},
		{"unsupported txt", "notes.txt", kindUnsupported},
		{"unsupported no extension", "README", kindUnsupported},
		{"unsupported doc (not docx)", "old.doc", kindUnsupported},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := classifyExtension(tt.filePath); got != tt.want {
				t.Errorf("classifyExtension(%q) = %d, want %d", tt.filePath, got, tt.want)
			}
		})
	}
}

// TestExtractorForUnsupported verifies that unsupported extensions produce a
// clear, actionable error naming the supported types.
func TestExtractorForUnsupported(t *testing.T) {
	tests := []struct {
		name     string
		filePath string
	}{
		{"txt file", "notes.txt"},
		{"csv file", "data.csv"},
		{"no extension", "Makefile"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ext, err := ExtractorFor(tt.filePath)
			if err == nil {
				t.Fatalf("ExtractorFor(%q) = %v, want error", tt.filePath, ext)
			}
			if !strings.Contains(err.Error(), "unsupported file type") {
				t.Errorf("error %q should explain the file type is unsupported", err)
			}
			if !strings.Contains(err.Error(), ".pdf") || !strings.Contains(err.Error(), ".docx") {
				t.Errorf("error %q should list the supported extensions", err)
			}
		})
	}
}

// TestParseDOCXBody verifies the WordprocessingML-to-text logic on hand-built
// XML, so it needs no real .docx file.
func TestParseDOCXBody(t *testing.T) {
	tests := []struct {
		name string
		xml  string
		want string
	}{
		{
			name: "single paragraph",
			xml:  `<w:document><w:body><w:p><w:r><w:t>Total revenue: 1,234</w:t></w:r></w:p></w:body></w:document>`,
			want: "Total revenue: 1,234",
		},
		{
			name: "two paragraphs separated by newline",
			xml:  `<w:body><w:p><w:r><w:t>Assets</w:t></w:r></w:p><w:p><w:r><w:t>Liabilities</w:t></w:r></w:p></w:body>`,
			want: "Assets\nLiabilities",
		},
		{
			name: "tab within a run",
			xml:  `<w:p><w:r><w:t>Q1</w:t><w:tab/><w:t>100</w:t></w:r></w:p>`,
			want: "Q1\t100",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseDOCXBody(strings.NewReader(tt.xml))
			if err != nil {
				t.Fatalf("parseDOCXBody returned error: %v", err)
			}
			if got != tt.want {
				t.Errorf("parseDOCXBody = %q, want %q", got, tt.want)
			}
		})
	}
}
