// Package ingestion handles turning source documents (PDF, DOCX, image) into
// plain text, then chunking and embedding that text for storage and retrieval.
package ingestion

import (
	"archive/zip"
	"context"
	"encoding/base64"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Extractor turns a document at filePath into plain extracted text.
type Extractor interface {
	Extract(ctx context.Context, filePath string) (string, error)
}

// PDFExtractor extracts text from PDFs by shelling out to `pdftotext`
// (poppler-utils). We deliberately avoid a pure-Go PDF library because
// extraction quality varies a lot between them, while pdftotext is reliable.
type PDFExtractor struct {
	binPath string
}

// NewPDFExtractor verifies that pdftotext is installed and returns an extractor
// bound to it.
func NewPDFExtractor() (*PDFExtractor, error) {
	binPath, err := exec.LookPath("pdftotext")
	if err != nil {
		return nil, fmt.Errorf("pdftotext not found — install poppler-utils: apt install poppler-utils: %w", err)
	}
	return &PDFExtractor{binPath: binPath}, nil
}

// Extract runs `pdftotext <file> -` and returns the extracted text on stdout.
func (e *PDFExtractor) Extract(ctx context.Context, filePath string) (string, error) {
	// "-" writes the extracted text to stdout instead of a file.
	cmd := exec.CommandContext(ctx, e.binPath, filePath, "-")
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to run pdftotext on %s: %w", filePath, err)
	}
	return string(out), nil
}

// DOCXExtractor extracts text from .docx files. A .docx is a zip archive whose
// word/document.xml holds the body; we unzip it and pull the text out of the
// <w:t> nodes with the standard library. This avoids github.com/unidoc/unioffice,
// which is commercially licensed (AGPLv3-or-commercial and requires a UniDoc
// license key at runtime) — overkill and licensing-encumbered for plain text
// extraction, so the archive/zip + encoding/xml approach is used instead.
type DOCXExtractor struct{}

// NewDOCXExtractor returns a DOCX extractor.
func NewDOCXExtractor() *DOCXExtractor {
	return &DOCXExtractor{}
}

// Extract reads word/document.xml from the .docx zip and concatenates the text
// runs, inserting a space between paragraphs so words don't run together.
func (e *DOCXExtractor) Extract(ctx context.Context, filePath string) (string, error) {
	zr, err := zip.OpenReader(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open docx %s as zip: %w", filePath, err)
	}
	defer zr.Close()

	var docXML *zip.File
	for _, f := range zr.File {
		if f.Name == "word/document.xml" {
			docXML = f
			break
		}
	}
	if docXML == nil {
		return "", fmt.Errorf("failed to extract docx %s: word/document.xml not found", filePath)
	}

	rc, err := docXML.Open()
	if err != nil {
		return "", fmt.Errorf("failed to open document.xml in %s: %w", filePath, err)
	}
	defer rc.Close()

	return parseDOCXBody(rc)
}

// parseDOCXBody streams the WordprocessingML body and reconstructs plain text:
// character data inside <w:t> runs becomes text, a <w:p> paragraph boundary
// becomes a newline, and a <w:tab> becomes a tab. Kept separate from Extract so
// the XML-to-text logic is unit-testable without building a real .docx.
func parseDOCXBody(r io.Reader) (string, error) {
	dec := xml.NewDecoder(r)
	var sb strings.Builder
	inText := false

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("failed to parse docx xml: %w", err)
		}

		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "t": // <w:t> — a run of literal text
				inText = true
			case "tab":
				sb.WriteByte('\t')
			case "br":
				sb.WriteByte('\n')
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "t":
				inText = false
			case "p": // paragraph boundary
				sb.WriteByte('\n')
			}
		case xml.CharData:
			if inText {
				sb.Write(t)
			}
		}
	}

	return strings.TrimSpace(sb.String()), nil
}

// ImageExtractor transcribes text (and describes tables/charts) from an image of
// a financial document by calling Claude with vision. Handles scanned pages and
// charts/tables that only exist in image form.
type ImageExtractor struct {
	client anthropic.Client
	model  anthropic.Model
}

const imageExtractionPrompt = "Transcribe all text and describe any tables/charts in this " +
	"financial document image. Preserve numbers exactly as shown."

// NewImageExtractor reads ANTHROPIC_API_KEY from the environment and returns an
// image extractor. Uses Claude Opus 4.8 (the SDK's most capable model constant);
// see the build summary for why this was chosen over the spec's placeholder ID.
func NewImageExtractor() (*ImageExtractor, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY environment variable is not set")
	}
	client := anthropic.NewClient(option.WithAPIKey(apiKey))
	return &ImageExtractor{
		client: client,
		model:  anthropic.ModelClaudeOpus4_8,
	}, nil
}

// Extract base64-encodes the image, sends it to Claude with the transcription
// prompt, and returns the concatenated text of the response.
func (e *ImageExtractor) Extract(ctx context.Context, filePath string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read image %s: %w", filePath, err)
	}

	mediaType, err := imageMediaType(filePath)
	if err != nil {
		return "", err
	}

	encoded := base64.StdEncoding.EncodeToString(data)

	msg, err := e.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     e.model,
		MaxTokens: 4096,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewImageBlockBase64(mediaType, encoded),
				anthropic.NewTextBlock(imageExtractionPrompt),
			),
		},
	})
	if err != nil {
		return "", fmt.Errorf("failed to transcribe image %s via Claude: %w", filePath, err)
	}

	var sb strings.Builder
	for _, block := range msg.Content {
		if text, ok := block.AsAny().(anthropic.TextBlock); ok {
			sb.WriteString(text.Text)
		}
	}
	return sb.String(), nil
}

// imageMediaType maps a supported image extension to its MIME type.
func imageMediaType(filePath string) (string, error) {
	switch strings.ToLower(filepath.Ext(filePath)) {
	case ".png":
		return "image/png", nil
	case ".jpg", ".jpeg":
		return "image/jpeg", nil
	default:
		return "", fmt.Errorf("unsupported image extension %q for %s", filepath.Ext(filePath), filePath)
	}
}

// extractorKind identifies which extractor an extension routes to. Kept as a
// pure classification so the routing logic can be unit-tested without
// constructing extractors (which require pdftotext / an API key).
type extractorKind int

const (
	kindUnsupported extractorKind = iota
	kindPDF
	kindDOCX
	kindImage
)

// classifyExtension maps a file path's extension to its extractor kind.
func classifyExtension(filePath string) extractorKind {
	switch strings.ToLower(filepath.Ext(filePath)) {
	case ".pdf":
		return kindPDF
	case ".docx":
		return kindDOCX
	case ".png", ".jpg", ".jpeg":
		return kindImage
	default:
		return kindUnsupported
	}
}

// ExtractorFor picks the right Extractor based on the file extension.
func ExtractorFor(filePath string) (Extractor, error) {
	switch classifyExtension(filePath) {
	case kindPDF:
		return NewPDFExtractor()
	case kindDOCX:
		return NewDOCXExtractor(), nil
	case kindImage:
		return NewImageExtractor()
	default:
		return nil, fmt.Errorf("unsupported file type %q: supported extensions are .pdf, .docx, .png, .jpg, .jpeg", filepath.Ext(filePath))
	}
}
