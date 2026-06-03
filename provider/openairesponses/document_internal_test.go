package openairesponses

import (
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

const testPDFData = "JVBERi0xLjQ=" // base64("%PDF-1.4")

// TestBuildRequestDocumentUnsupported verifies that the Responses API adapter
// fails loudly on document content instead of silently dropping it.
func TestBuildRequestDocumentUnsupported(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5.1"}, providerName: "openai_responses"}

	req := llm.Request{
		MaxTokens: 1024,
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentPart{
					{
						Type:     llm.ContentDocument,
						Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: testPDFData},
					},
				},
			},
		},
	}

	_, err := c.buildRequest(req, false)
	if err == nil {
		t.Fatal("buildRequest with document content = nil error, want explicit error")
	}
	if !strings.Contains(err.Error(), "document") {
		t.Errorf("error = %q, want it to mention 'document'", err.Error())
	}
}
