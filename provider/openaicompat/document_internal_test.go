package openaicompat

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

const testPDFData = "JVBERi0xLjQ=" // base64("%PDF-1.4")

// TestBuildRequestDocumentAsImageURL verifies that a ContentDocument part is
// serialized as an image_url data URI in the multipart content array — the same
// shape vision models such as Grok expect for PDFs.
func TestBuildRequestDocumentAsImageURL(t *testing.T) {
	c := &Client{providerName: "xai", config: llm.Config{Model: "grok-2-vision-1212"}}

	req := llm.Request{
		MaxTokens: 4096,
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

	contentArray, ok := apiReq.Messages[0].Content.([]any)
	if !ok {
		t.Fatalf("Content is %T, want []any (multipart)", apiReq.Messages[0].Content)
	}

	wantURI := "data:application/pdf;base64," + testPDFData
	var foundDoc bool
	for _, item := range contentArray {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if m["type"] != "image_url" {
			continue
		}
		iu, ok := m["image_url"].(map[string]string)
		if !ok {
			t.Fatalf("image_url is %T, want map[string]string", m["image_url"])
		}
		if iu["url"] == wantURI {
			foundDoc = true
		}
	}
	if !foundDoc {
		t.Errorf("document image_url data URI %q not found in content array %+v", wantURI, contentArray)
	}
}

// TestBuildRequestDocumentInvalidMediaType verifies that an unsupported document
// media type is rejected at request-build time.
func TestBuildRequestDocumentInvalidMediaType(t *testing.T) {
	c := &Client{providerName: "xai", config: llm.Config{Model: "grok-2-vision-1212"}}

	req := llm.Request{
		MaxTokens: 4096,
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
