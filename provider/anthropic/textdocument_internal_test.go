package anthropic

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestTextDocumentOnTheWire(t *testing.T) {
	doc := llm.NewTextDocument("Reserved names", "example.com is reserved for documentation.")
	doc.Context = "Public registry note."
	doc.CacheControl, doc.CacheTTL = true, llm.CacheTTL1h
	msg := llm.Message{Role: llm.RoleUser, Content: []llm.ContentPart{doc, {Type: llm.ContentText, Text: "What is it for?"}}}

	c := &Client{config: llm.Config{Model: "claude-haiku-5-5", MaxTokens: 512}, providerName: "anthropic"}
	data, err := c.BuildMessageBatchParams(llm.Request{Messages: []llm.Message{msg}})
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(data, &body); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"type":          "document",
		"source":        map[string]any{"type": "text", "media_type": "text/plain", "data": "example.com is reserved for documentation."},
		"title":         "Reserved names",
		"context":       "Public registry note.",
		"cache_control": map[string]any{"type": "ephemeral", "ttl": "1h"},
	}
	if got := body.Messages[0].Content[0]; !reflect.DeepEqual(got, want) {
		t.Errorf("document block\n got %v\nwant %v", got, want)
	}

	// The neutral part survives a JSON round trip unchanged.
	raw, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	var back llm.Message
	if err := json.Unmarshal(raw, &back); err != nil || !reflect.DeepEqual(back, msg) {
		t.Errorf("round trip: %+v, %v", back, err)
	}
}

func TestTextDocumentValidation(t *testing.T) {
	part := func(mediaType, data string) llm.ContentPart {
		return llm.ContentPart{Type: llm.ContentDocument, Document: &llm.DocumentSource{Type: llm.DocumentSourceText, MediaType: mediaType, Data: data}}
	}
	for name, p := range map[string]llm.ContentPart{
		"empty":         part("text/plain", ""),
		"blank":         part("text/plain", " \n"),
		"media type":    part("text/markdown", "x"),
		"invalid UTF-8": part("text/plain", "\xff\xfe"),
	} {
		if err := llm.ValidateDocumentSource(p); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	if err := llm.ValidateDocumentSource(part("text/plain", "ok")); err != nil {
		t.Errorf("valid text document: %v", err)
	}
	// A PDF source keeps its checks, and a text body is no PDF.
	pdf := llm.ContentPart{Type: llm.ContentDocument, Document: &llm.DocumentSource{Type: llm.DocumentSourceBase64, MediaType: "text/plain", Data: "b2s="}}
	if err := llm.ValidateDocumentSource(pdf); err == nil || !strings.Contains(err.Error(), "media type") {
		t.Errorf("base64 text/plain: %v", err)
	}
}
