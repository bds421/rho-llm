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

func TestBuildRequestRefusesTextDocument(t *testing.T) {
	c := &Client{providerName: "gemini"}
	req := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{llm.NewTextDocument("", "plain text")}}}}
	if _, err := c.buildRequest(req); err == nil {
		t.Error("text document accepted; it would be sent as base64 data")
	}
}

func TestUnsupportedAnthropicFieldsRefused(t *testing.T) {
	c := &Client{providerName: "gemini"}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}
	for name, req := range map[string]llm.Request{
		"effort":    {Effort: llm.EffortLow, Messages: msgs},
		"container": {ContainerID: "container_1", Messages: msgs},
	} {
		if _, err := c.buildRequest(req); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	for raw, want := range map[string]string{"SAFETY": llm.StopRefusal, "PROHIBITED_CONTENT": llm.StopRefusal, "RECITATION": llm.StopRefusal, "MALFORMED_FUNCTION_CALL": llm.StopError} {
		if got := normalizeStopReason(raw); got != want {
			t.Errorf("%s -> %s, want %s", raw, got, want)
		}
	}
}
