package openaicompat

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/bds421/rho-llm"
)

const testPDFData = "JVBERi0xLjQ=" // base64("%PDF-1.4")

func pdfRequest(parts ...llm.ContentPart) llm.Request {
	return llm.Request{MaxTokens: 4096, Messages: []llm.Message{{Role: llm.RoleUser, Content: append(
		[]llm.ContentPart{{Type: llm.ContentText, Text: "Extract this invoice."}}, parts...)}}}
}

func pdfPart(title string) llm.ContentPart {
	return llm.ContentPart{Type: llm.ContentDocument, Title: title,
		Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: testPDFData}}
}

// contentParts returns the multipart content of the only message.
func contentParts(t *testing.T, apiReq openaiRequest) []map[string]any {
	t.Helper()
	if len(apiReq.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1", len(apiReq.Messages))
	}
	arr, ok := apiReq.Messages[0].Content.([]any)
	if !ok {
		t.Fatalf("Content is %T, want []any (multipart)", apiReq.Messages[0].Content)
	}
	out := make([]map[string]any, 0, len(arr))
	for _, item := range arr {
		out = append(out, item.(map[string]any))
	}
	return out
}

// A PDF goes out in the host's own part. Every host rejects it as an image_url
// data URI (live 2026-11-18: OpenAI 400 invalid_image_format, xAI 400
// invalid_image, Mistral 422), which is what the adapter used to send.
func TestBuildRequestDocumentPartPerHost(t *testing.T) {
	uri := "data:application/pdf;base64," + testPDFData
	for _, tc := range []struct {
		provider, baseURL, wantType string
	}{
		{"openai", "", "file"},
		{"openrouter", "", "file"},
		{"together", "", "file"},
		{"custom", "https://proxy.example/v1", "file"}, // unknown host: the OpenAI shape
		{"mistral", "", "document_url"},
		{"my-mistral", "https://api.mistral.ai/v1", "document_url"}, // detected by BaseURL
	} {
		c := &Client{providerName: tc.provider, baseURL: tc.baseURL, config: llm.Config{Model: "m"}}
		apiReq, err := c.buildRequest(pdfRequest(pdfPart("invoice.pdf"), pdfPart("")), false)
		if err != nil {
			t.Fatalf("%s: %v", tc.provider, err)
		}
		parts := contentParts(t, apiReq)
		if len(parts) != 3 || parts[0]["type"] != "text" {
			t.Fatalf("%s: parts %+v", tc.provider, parts)
		}
		for k, part := range parts[1:] {
			if part["type"] != tc.wantType {
				t.Errorf("%s: document %d type %v, want %s", tc.provider, k, part["type"], tc.wantType)
				continue
			}
			if tc.wantType == "document_url" {
				if part["document_url"] != uri {
					t.Errorf("%s: document_url %v", tc.provider, part["document_url"])
				}
				continue
			}
			f, ok := part["file"].(map[string]string)
			wantName := []string{"invoice.pdf", "document-2.pdf"}[k]
			if !ok || f["file_data"] != uri || f["filename"] != wantName {
				t.Errorf("%s: file %+v, want filename %q and the data URL", tc.provider, part["file"], wantName)
			}
		}
	}
}

// Images keep their image_url part next to a PDF.
func TestBuildRequestImageStaysImageURLNextToDocument(t *testing.T) {
	c := &Client{providerName: "mistral", config: llm.Config{Model: "m"}}
	img := llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "iVBORw0KGgo="}}
	apiReq, err := c.buildRequest(pdfRequest(img, pdfPart("")), false)
	if err != nil {
		t.Fatal(err)
	}
	parts := contentParts(t, apiReq)
	if parts[1]["type"] != "image_url" || parts[2]["type"] != "document_url" {
		t.Errorf("parts %+v", parts)
	}
}

// xAI Chat Completions takes no files ("Please use /v1/responses instead"):
// the PDF is refused before the request, by preset, alias or BaseURL.
func TestBuildRequestDocumentRefusedOnXAI(t *testing.T) {
	for _, c := range []*Client{
		{providerName: "xai", config: llm.Config{Model: "grok-4.7"}},
		{providerName: "grok", config: llm.Config{Model: "grok-4.7"}},
		{providerName: "custom", baseURL: "https://api.x.ai/v1", config: llm.Config{Model: "grok-4.7"}},
	} {
		_, err := c.buildRequest(pdfRequest(pdfPart("")), false)
		if err == nil || !strings.Contains(err.Error(), "Responses") {
			t.Errorf("%s: err %v, want a refusal naming the Responses API", c.providerName, err)
		}
	}
	// An image alone still goes through on xAI.
	c := &Client{providerName: "xai", config: llm.Config{Model: "grok-4.7"}}
	img := llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "iVBORw0KGgo="}}
	if _, err := c.buildRequest(pdfRequest(img), false); err != nil {
		t.Errorf("image refused on xai: %v", err)
	}
}

// Through the public client: the refusal happens before any HTTP request.
func TestXAIDocumentRefusedBeforeRequest(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.Error(w, "unexpected", http.StatusTeapot)
	}))
	defer srv.Close()
	c, err := New(llm.Config{Provider: "xai", Model: "grok-4.7", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := c.Complete(context.Background(), pdfRequest(pdfPart(""))); err == nil {
		t.Error("PDF accepted on xai")
	}
	for _, err := range c.Stream(context.Background(), pdfRequest(pdfPart(""))) {
		if err == nil {
			t.Error("stream: PDF accepted on xai")
		}
	}
	if hits.Load() != 0 {
		t.Errorf("%d requests reached the server", hits.Load())
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

func TestBuildRequestRefusesTextDocument(t *testing.T) {
	c := &Client{providerName: "xai", config: llm.Config{Model: "grok-2-vision-1212"}}
	req := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{llm.NewTextDocument("", "plain text")}}}}
	if _, err := c.buildRequest(req, false); err == nil {
		t.Error("text document accepted; it would be sent as a data URI")
	}
}

func TestUnsupportedAnthropicFieldsRefused(t *testing.T) {
	c := &Client{providerName: "xai", config: llm.Config{Model: "grok-2-vision-1212"}}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}
	for name, req := range map[string]llm.Request{
		"effort":    {Effort: llm.EffortLow, Messages: msgs},
		"container": {ContainerID: "container_1", Messages: msgs},
	} {
		if _, err := c.buildRequest(req, false); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	if got := normalizeStopReason("content_filter"); got != llm.StopRefusal {
		t.Errorf("content_filter -> %s", got)
	}
}
