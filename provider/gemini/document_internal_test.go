package gemini

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

const testPDFData = "JVBERi0xLjQ=" // base64("%PDF-1.4")

// TestBuildRequestDocumentInlineData verifies that a ContentDocument part is
// serialized as a Gemini inlineData part (native PDF parsing), alongside text.
func TestBuildRequestDocumentInlineData(t *testing.T) {
	c := &Client{providerName: "gemini"}

	req := llm.Request{
		MaxTokens: 1024,
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{Type: llm.ContentText, Text: "Extract this invoice."},
					{
						Type:     llm.ContentDocument,
						Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: testPDFData},
					},
				},
			},
		},
	}

	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if len(apiReq.Contents) != 1 {
		t.Fatalf("len(Contents) = %d, want 1", len(apiReq.Contents))
	}
	parts := apiReq.Contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2 (text + document)", len(parts))
	}

	var inline *geminiInlineData
	for _, p := range parts {
		if p.InlineData != nil {
			inline = p.InlineData
		}
	}
	if inline == nil {
		t.Fatal("no inlineData part found for document")
	}
	if inline.MimeType != "application/pdf" {
		t.Errorf("inlineData.MimeType = %q, want application/pdf", inline.MimeType)
	}
	if inline.Data != testPDFData {
		t.Errorf("inlineData.Data = %q, want %q", inline.Data, testPDFData)
	}
}

// TestBuildRequestDocumentInvalidMediaType verifies that an unsupported document
// media type is rejected at request-build time.
func TestBuildRequestDocumentInvalidMediaType(t *testing.T) {
	c := &Client{providerName: "gemini"}

	req := llm.Request{
		MaxTokens: 1024,
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{
						Type:     llm.ContentDocument,
						Document: &llm.DocumentSource{Type: "base64", MediaType: "text/plain", Data: testPDFData},
					},
				},
			},
		},
	}

	if _, err := c.buildRequest(req); err == nil {
		t.Error("buildRequest with unsupported document media type = nil error, want error")
	}
}
