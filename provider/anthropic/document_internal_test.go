package anthropic

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

const testPDFData = "JVBERi0xLjQ=" // base64("%PDF-1.4")

// TestBuildRequestDocumentBlock verifies that a ContentDocument part is
// serialized as an Anthropic "document" content block (native PDF parsing).
func TestBuildRequestDocumentBlock(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-4-5"}, providerName: "anthropic"}

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

	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if len(apiReq.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1", len(apiReq.Messages))
	}

	var docBlock map[string]any
	for _, raw := range apiReq.Messages[0].Content {
		block, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if block["type"] == "document" {
			docBlock = block
		}
	}
	if docBlock == nil {
		t.Fatalf("no document block found in content %+v", apiReq.Messages[0].Content)
	}

	source, ok := docBlock["source"].(map[string]any)
	if !ok {
		t.Fatalf("document source is %T, want map[string]any", docBlock["source"])
	}
	if source["media_type"] != "application/pdf" {
		t.Errorf("source.media_type = %v, want application/pdf", source["media_type"])
	}
	if source["data"] != testPDFData {
		t.Errorf("source.data = %v, want %q", source["data"], testPDFData)
	}
	if source["type"] != "base64" {
		t.Errorf("source.type = %v, want base64", source["type"])
	}
}

// TestBuildRequestDocumentInvalidMediaType verifies that an unsupported document
// media type is rejected at request-build time.
func TestBuildRequestDocumentInvalidMediaType(t *testing.T) {
	c := &Client{config: llm.Config{Model: "claude-opus-4-5"}, providerName: "anthropic"}

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

	if _, err := c.buildRequest(req, false); err == nil {
		t.Error("buildRequest with unsupported document media type = nil error, want error")
	}
}
