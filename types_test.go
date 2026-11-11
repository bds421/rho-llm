package llm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

func TestNewAssistantMessagePanicsOnNil(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewAssistantMessage(nil) did not panic")
		}
	}()
	_ = llm.NewAssistantMessage(nil)
}

func TestAudioMediaTypeFromSignature(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
		want string
	}{
		{"wav", []byte("RIFF\x00\x00\x00\x00WAVEfmt "), "audio/wav"},
		{"riff but avi", []byte("RIFF\x00\x00\x00\x00AVI LIST"), ""},
		{"flac", []byte("fLaC\x00\x00"), "audio/flac"},
		{"mp3 id3", []byte("ID3\x04\x00"), "audio/mpeg"},
		{"mp3 frame sync", []byte{0xff, 0xfb, 0x90, 0x00}, "audio/mpeg"},
		{"aac adts mpeg-4", []byte{0xff, 0xf1, 0x50, 0x80}, "audio/aac"},
		{"aac adts mpeg-2", []byte{0xff, 0xf9, 0x50, 0x80}, "audio/aac"},
		{"ogg", []byte("OggS\x00\x02"), "audio/ogg"},
		{"m4a", []byte("\x00\x00\x00\x20ftypM4A "), "audio/mp4"},
		{"aiff", []byte("FORM\x00\x00\x00\x00AIFFCOMM"), "audio/aiff"},
		{"aifc", []byte("FORM\x00\x00\x00\x00AIFCFVER"), "audio/aiff"},
		{"form but not audio", []byte("FORM\x00\x00\x00\x00ILBM"), ""},
		{"webm", []byte{0x1a, 0x45, 0xdf, 0xa3, 0x01}, "audio/webm"},
		{"png", []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}, ""},
		{"text", []byte("hello world, not audio"), ""},
		{"nil", nil, ""},
		{"one byte", []byte{0xff}, ""},
		{"truncated riff", []byte("RIFF\x00\x00"), ""},
		{"truncated form", []byte("FORM\x00\x00\x00\x00AI"), ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := llm.AudioMediaTypeFromSignature(tc.data); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// validPDFData is a tiny, non-empty base64 payload ("%PDF-1.4").
const validPDFData = "JVBERi0xLjQ="

func TestValidateDocumentSource(t *testing.T) {
	tests := []struct {
		name    string
		part    llm.ContentPart
		wantErr bool
	}{
		{
			name: "valid pdf",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: validPDFData},
			},
			wantErr: false,
		},
		{
			name:    "nil document",
			part:    llm.ContentPart{Type: llm.ContentDocument},
			wantErr: true,
		},
		{
			name: "empty data",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: ""},
			},
			wantErr: true,
		},
		{
			name: "unsupported media type",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "base64", MediaType: "image/png", Data: validPDFData},
			},
			wantErr: true,
		},
		{
			name: "unsupported source type",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "url", MediaType: "application/pdf", Data: validPDFData},
			},
			wantErr: true,
		},
		{
			name: "non-canonical base64",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: "JVBERi0"},
			},
			wantErr: true,
		},
		{
			name: "mislabeled bytes",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: "iVBORw0KGgo="},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := llm.ValidateDocumentSource(tc.part)
			if tc.wantErr && err == nil {
				t.Errorf("ValidateDocumentSource(%s) = nil, want error", tc.name)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("ValidateDocumentSource(%s) = %v, want nil", tc.name, err)
			}
		})
	}
}

func TestNewDocumentMessage(t *testing.T) {
	msg := llm.NewDocumentMessage(llm.RoleUser, "application/pdf", validPDFData)

	if msg.Role != llm.RoleUser {
		t.Errorf("Role = %q, want %q", msg.Role, llm.RoleUser)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
	}
	part := msg.Content[0]
	if part.Type != llm.ContentDocument {
		t.Errorf("Type = %q, want %q", part.Type, llm.ContentDocument)
	}
	if part.Document == nil {
		t.Fatal("Document is nil")
	}
	if part.Document.Type != "base64" {
		t.Errorf("Document.Type = %q, want base64", part.Document.Type)
	}
	if part.Document.MediaType != "application/pdf" {
		t.Errorf("Document.MediaType = %q, want application/pdf", part.Document.MediaType)
	}
	if part.Document.Data != validPDFData {
		t.Errorf("Document.Data = %q, want %q", part.Document.Data, validPDFData)
	}

	// The constructed message must pass validation.
	if err := llm.ValidateDocumentSource(part); err != nil {
		t.Errorf("ValidateDocumentSource on constructed message: %v", err)
	}
}

// ValidateImageSource must reject every structurally-malformed image part (the
// document validator had a direct test; the image one did not).
func TestValidateImageSourceRejectsMalformed(t *testing.T) {
	good := llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "iVBORw0KGgo="}}
	if err := llm.ValidateImageSource(good); err != nil {
		t.Fatalf("a valid image part was rejected: %v", err)
	}

	bad := map[string]*llm.ImageSource{
		"nil source":             nil,
		"empty data":             {Type: "base64", MediaType: "image/png", Data: ""},
		"unsupported media type": {Type: "base64", MediaType: "image/tiff", Data: "abc"},
		"empty media type":       {Type: "base64", MediaType: "", Data: "abc"},
		"non-base64 source type": {Type: "url", MediaType: "image/png", Data: "abc"},
		"non-canonical base64":   {Type: "base64", MediaType: "image/png", Data: "iVBORw0KGgo"},
		"unknown signature":      {Type: "base64", MediaType: "image/png", Data: "bm90LWltYWdl"},
		"mislabeled jpeg":        {Type: "base64", MediaType: "image/png", Data: "/9j/4A=="},
	}
	for name, src := range bad {
		part := llm.ContentPart{Type: llm.ContentImage, Source: src}
		if err := llm.ValidateImageSource(part); err == nil {
			t.Errorf("%s: ValidateImageSource accepted malformed input", name)
		}
	}
}

func TestValidateImageSourceAcceptsEverySupportedSignature(t *testing.T) {
	for mediaType, data := range map[string]string{
		"image/png":  "iVBORw0KGgo=",
		"image/jpeg": "/9j/4A==",
		"image/gif":  "R0lGODdh",
		"image/webp": "UklGRgQAAABXRUJQ",
	} {
		part := llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{
			Type: "base64", MediaType: mediaType, Data: data,
		}}
		if err := llm.ValidateImageSource(part); err != nil {
			t.Errorf("%s: %v", mediaType, err)
		}
	}
}

// TestNewTextMessage verifies text message construction.
func TestNewTextMessage(t *testing.T) {
	msg := llm.NewTextMessage("user", "hello")
	if msg.Role != llm.RoleUser {
		t.Errorf("Expected role 'user', got %q", msg.Role)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("Expected 1 content part, got %d", len(msg.Content))
	}
	if msg.Content[0].Type != llm.ContentText || msg.Content[0].Text != "hello" {
		t.Errorf("Unexpected content: %+v", msg.Content[0])
	}
}

// TestNewSystemMessage verifies system message construction.
func TestNewSystemMessage(t *testing.T) {
	msg := llm.NewSystemMessage("You are a helpful assistant.")
	if msg.Role != llm.RoleSystem {
		t.Errorf("Expected role 'system', got %q", msg.Role)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("Expected 1 content part, got %d", len(msg.Content))
	}
	if msg.Content[0].Type != llm.ContentText || msg.Content[0].Text != "You are a helpful assistant." {
		t.Errorf("Unexpected content: %+v", msg.Content[0])
	}
}

// TestNewToolResultMessage verifies tool result message construction.
func TestNewToolResultMessage(t *testing.T) {
	msg := llm.NewToolResultMessage("tool-123", "result data", false)
	if msg.Role != llm.RoleUser {
		t.Errorf("Expected role 'user', got %q", msg.Role)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("Expected 1 content part, got %d", len(msg.Content))
	}
	part := msg.Content[0]
	if part.Type != llm.ContentToolResult {
		t.Errorf("Expected type 'tool_result', got %q", part.Type)
	}
	if part.ToolResultID != "tool-123" {
		t.Errorf("Expected tool_use_id 'tool-123', got %q", part.ToolResultID)
	}
}

// TestContentImageValidation verifies validation across all 3 providers using
// table-driven tests: nil source, empty data, invalid media type, invalid source type.
func TestContentImageValidation(t *testing.T) {
	providers := []struct {
		name     string
		provider string
		model    string
	}{
		{"anthropic", "anthropic", "claude-sonnet-4-6"},
		{"gemini", "gemini", "gemini-2.5-flash"},
		{"openai", "openai", "gpt-4"},
	}

	cases := []struct {
		name    string
		part    llm.ContentPart
		wantErr string
	}{
		{
			name:    "nil source",
			part:    llm.ContentPart{Type: llm.ContentImage, Source: nil},
			wantErr: "image source must not be nil",
		},
		{
			name:    "empty data",
			part:    llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: ""}},
			wantErr: "image data must not be empty",
		},
		{
			name:    "invalid media type",
			part:    llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "text/plain", Data: "abc"}},
			wantErr: "unsupported image media type",
		},
		{
			name:    "invalid source type",
			part:    llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "url", MediaType: "image/png", Data: "abc"}},
			wantErr: "unsupported image source type",
		},
	}

	for _, prov := range providers {
		for _, tc := range cases {
			t.Run(prov.name+"/"+tc.name, func(t *testing.T) {
				cfg := llm.Config{
					Provider:          prov.provider,
					Model:             prov.model,
					ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityVision),
					APIKey:            "test-key",
					BaseURL:           "http://localhost:1",
				}
				client, err := llm.NewClient(cfg)
				if err != nil {
					t.Fatalf("NewClient: %v", err)
				}
				defer client.Close()

				req := llm.Request{
					Messages: []llm.Message{{
						Role:    llm.RoleUser,
						Content: []llm.ContentPart{tc.part},
					}},
				}
				_, err = client.Complete(context.Background(), req)
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("expected error containing %q, got: %v", tc.wantErr, err)
				}
			})
		}
	}
}

// TestContentImageAnthropic verifies the Anthropic wire format for image content.
func TestContentImageAnthropic(t *testing.T) {
	imgData := loadTestImage(t, "red.png")

	var captured json.RawMessage
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured = body
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"I see red"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","usage":{"input_tokens":100,"output_tokens":10}}`)
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider: "anthropic",
		Model:    "claude-sonnet-4-6",
		APIKey:   "test-key",
		BaseURL:  srv.URL,
	}
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	req := llm.Request{
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			Content: []llm.ContentPart{
				{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: imgData}},
			},
		}},
	}

	resp, err := client.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "I see red" {
		t.Errorf("unexpected content: %s", resp.Content)
	}

	// Verify wire format
	var wire struct {
		Messages []struct {
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(captured, &wire); err != nil {
		t.Fatalf("unmarshal captured request: %v", err)
	}

	if len(wire.Messages) == 0 || len(wire.Messages[0].Content) == 0 {
		t.Fatal("no content blocks in wire request")
	}

	var block struct {
		Type   string `json:"type"`
		Source struct {
			Type      string `json:"type"`
			MediaType string `json:"media_type"`
			Data      string `json:"data"`
		} `json:"source"`
	}
	if err := json.Unmarshal(wire.Messages[0].Content[0], &block); err != nil {
		t.Fatalf("unmarshal content block: %v", err)
	}

	if block.Type != "image" {
		t.Errorf("block type = %s, want image", block.Type)
	}
	if block.Source.Type != "base64" {
		t.Errorf("source type = %s, want base64", block.Source.Type)
	}
	if block.Source.MediaType != "image/png" {
		t.Errorf("media_type = %s, want image/png", block.Source.MediaType)
	}
	if block.Source.Data != imgData {
		t.Error("image data mismatch")
	}
}

// TestContentImageGemini verifies the Gemini wire format for image content.
func TestContentImageGemini(t *testing.T) {
	imgData := loadTestImage(t, "red.png")

	var captured json.RawMessage
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured = body
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"I see red"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":10,"totalTokenCount":110}}`)
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider: "gemini",
		Model:    "gemini-2.5-flash",
		APIKey:   "test-key",
		BaseURL:  srv.URL,
	}
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	req := llm.Request{
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			Content: []llm.ContentPart{
				{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: imgData}},
			},
		}},
	}

	resp, err := client.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "I see red" {
		t.Errorf("unexpected content: %s", resp.Content)
	}

	// Verify wire format
	var wire struct {
		Contents []struct {
			Parts []struct {
				InlineData *struct {
					MimeType string `json:"mimeType"`
					Data     string `json:"data"`
				} `json:"inlineData,omitempty"`
			} `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(captured, &wire); err != nil {
		t.Fatalf("unmarshal captured request: %v", err)
	}

	if len(wire.Contents) == 0 || len(wire.Contents[0].Parts) == 0 {
		t.Fatal("no parts in wire request")
	}

	part := wire.Contents[0].Parts[0]
	if part.InlineData == nil {
		t.Fatal("inlineData is nil")
	}
	if part.InlineData.MimeType != "image/png" {
		t.Errorf("mimeType = %s, want image/png", part.InlineData.MimeType)
	}
	if part.InlineData.Data != imgData {
		t.Error("image data mismatch")
	}
}

// TestContentImageOpenAI verifies the OpenAI wire format: content switches from
// string to array when images are present.
func TestContentImageOpenAI(t *testing.T) {
	imgData := loadTestImage(t, "blue.jpg")

	var captured json.RawMessage
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured = body
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"I see blue"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10}}`)
	}))
	defer srv.Close()

	cfg := llm.Config{
		Provider:          "openai",
		Model:             "gpt-4",
		ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityVision),
		APIKey:            "test-key",
		BaseURL:           srv.URL,
	}
	client, err := llm.NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	req := llm.Request{
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			Content: []llm.ContentPart{
				{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/jpeg", Data: imgData}},
			},
		}},
	}

	resp, err := client.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "I see blue" {
		t.Errorf("unexpected content: %s", resp.Content)
	}

	// Verify wire format — content must be an array, not a string
	var wire struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(captured, &wire); err != nil {
		t.Fatalf("unmarshal captured request: %v", err)
	}

	// Find the user message
	var userContent json.RawMessage
	for _, m := range wire.Messages {
		if m.Role == "user" {
			userContent = m.Content
			break
		}
	}
	if userContent == nil {
		t.Fatal("no user message in wire request")
	}

	// Content must be an array (not a string)
	var contentArr []struct {
		Type     string `json:"type"`
		ImageURL *struct {
			URL string `json:"url"`
		} `json:"image_url,omitempty"`
	}
	if err := json.Unmarshal(userContent, &contentArr); err != nil {
		t.Fatalf("content is not an array: %v", err)
	}

	found := false
	for _, item := range contentArr {
		if item.Type == "image_url" && item.ImageURL != nil {
			wantPrefix := "data:image/jpeg;base64,"
			if !strings.HasPrefix(item.ImageURL.URL, wantPrefix) {
				t.Errorf("image_url prefix = %s..., want %s...", item.ImageURL.URL[:min(len(item.ImageURL.URL), 30)], wantPrefix)
			}
			found = true
		}
	}
	if !found {
		t.Error("no image_url block found in content array")
	}
}

// TestContentImageMixedMessage verifies text + image in one message across all 3 providers.
func TestContentImageMixedMessage(t *testing.T) {
	imgData := loadTestImage(t, "red.png")

	providers := []struct {
		name     string
		provider string
		model    string
		response string
	}{
		{
			"anthropic", "anthropic", "claude-sonnet-4-6",
			`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"mixed ok"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","usage":{"input_tokens":100,"output_tokens":5}}`,
		},
		{
			"gemini", "gemini", "gemini-2.5-flash",
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"mixed ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"totalTokenCount":105}}`,
		},
		{
			"openai", "openai", "gpt-4",
			`{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"mixed ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":5}}`,
		},
	}

	for _, prov := range providers {
		t.Run(prov.name, func(t *testing.T) {
			var captured json.RawMessage
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				captured = body
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, prov.response)
			}))
			defer srv.Close()

			cfg := llm.Config{
				Provider:          prov.provider,
				Model:             prov.model,
				ModelCapabilities: llm.Capabilities(llm.CapabilityChat, llm.CapabilityVision),
				APIKey:            "test-key",
				BaseURL:           srv.URL,
			}
			client, err := llm.NewClient(cfg)
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			defer client.Close()

			req := llm.Request{
				Messages: []llm.Message{{
					Role: llm.RoleUser,
					Content: []llm.ContentPart{
						{Type: llm.ContentText, Text: "What is in this image?"},
						{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: imgData}},
					},
				}},
			}

			resp, err := client.Complete(context.Background(), req)
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if resp.Content != "mixed ok" {
				t.Errorf("unexpected content: %s", resp.Content)
			}

			// Verify the wire request was captured (basic sanity)
			if len(captured) == 0 {
				t.Fatal("no request captured")
			}
		})
	}
}

// TestNewImageMessage verifies the NewImageMessage helper function.
func TestNewImageMessage(t *testing.T) {
	msg := llm.NewImageMessage(llm.RoleUser, "image/png", "base64data")

	if msg.Role != llm.RoleUser {
		t.Errorf("role = %s, want user", msg.Role)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content parts = %d, want 1", len(msg.Content))
	}

	part := msg.Content[0]
	if part.Type != llm.ContentImage {
		t.Errorf("type = %s, want image", part.Type)
	}
	if part.Source == nil {
		t.Fatal("source is nil")
	}
	if part.Source.Type != "base64" {
		t.Errorf("source type = %s, want base64", part.Source.Type)
	}
	if part.Source.MediaType != "image/png" {
		t.Errorf("media type = %s, want image/png", part.Source.MediaType)
	}
	if part.Source.Data != "base64data" {
		t.Errorf("data = %s, want base64data", part.Source.Data)
	}
}

// TestNewAssistantMessage verifies assistant message construction including tool calls.
func TestNewAssistantMessage(t *testing.T) {
	resp := &llm.Response{
		Content: "Here is the result:",
		ToolCalls: []llm.ToolCall{
			{ID: "call_1", Name: "get_weather", Input: map[string]interface{}{"city": "Berlin"}},
			{ID: "call_2", Name: "get_time", Input: map[string]interface{}{"tz": "UTC"}, ThoughtSignature: "sig123"},
		},
	}

	msg := llm.NewAssistantMessage(resp)

	if msg.Role != llm.RoleAssistant {
		t.Errorf("Role = %q, want assistant", msg.Role)
	}
	if len(msg.Content) != 3 {
		t.Fatalf("Content parts = %d, want 3 (1 text + 2 tool_use)", len(msg.Content))
	}

	// Text part
	if msg.Content[0].Type != llm.ContentText || msg.Content[0].Text != "Here is the result:" {
		t.Errorf("content[0] = %+v, want text part", msg.Content[0])
	}

	// Tool use parts
	if msg.Content[1].Type != llm.ContentToolUse || msg.Content[1].ToolName != "get_weather" {
		t.Errorf("content[1] = %+v, want get_weather tool_use", msg.Content[1])
	}
	if msg.Content[2].ThoughtSignature != "sig123" {
		t.Errorf("content[2].ThoughtSignature = %q, want sig123", msg.Content[2].ThoughtSignature)
	}
}

// TestNewAssistantMessageTextOnly verifies message with no tool calls.
func TestNewAssistantMessageTextOnly(t *testing.T) {
	resp := &llm.Response{Content: "Just text"}
	msg := llm.NewAssistantMessage(resp)
	if len(msg.Content) != 1 || msg.Content[0].Type != llm.ContentText {
		t.Errorf("expected single text part, got %+v", msg.Content)
	}
}

// TestNewAssistantMessageEmpty verifies empty response produces empty content.
func TestNewAssistantMessageEmpty(t *testing.T) {
	resp := &llm.Response{}
	msg := llm.NewAssistantMessage(resp)
	if len(msg.Content) != 0 {
		t.Errorf("expected zero content parts, got %d", len(msg.Content))
	}
}

// TestNewAssistantMessageNilPanics verifies nil response panics with clear message.
func TestNewAssistantMessageNilPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for nil resp")
		}
		msg := fmt.Sprint(r)
		if !strings.Contains(msg, "must not be nil") {
			t.Errorf("panic message = %q, want 'must not be nil'", msg)
		}
	}()
	llm.NewAssistantMessage(nil)
}

func TestSetPartsDerivesFlatFields(t *testing.T) {
	var r llm.Response
	r.SetParts([]llm.ContentPart{
		{Type: llm.ContentThinking, Thinking: "", ThinkingSignature: "sig"},
		{Type: llm.ContentText, Text: "a"},
		{Type: llm.ContentServerToolUse, ToolUseID: "srv_1", ToolName: "web_search"},
		{Type: llm.ContentText, Text: "b"},
		{Type: llm.ContentToolUse, ToolUseID: "t1", ToolName: "lookup", ToolInput: map[string]any{"q": "x"}},
		{Type: llm.ContentThinking, Thinking: "r", Redacted: true},
	})
	if r.Content != "ab" || r.Thinking != "r" || r.ThinkingSignature != "sig" || !r.ThinkingRedacted ||
		len(r.ToolCalls) != 1 || r.ToolCalls[0].ID != "t1" || len(r.Parts) != 6 {
		t.Errorf("%+v", r)
	}
}

func TestToolResultTextDegrade(t *testing.T) {
	// Rich tool result: text + image. Providers without image support get text only.
	part := llm.ContentPart{
		Type:         llm.ContentToolResult,
		ToolResultID: "c1",
		ToolResultParts: []llm.ContentPart{
			{Type: llm.ContentText, Text: "line1"},
			{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}},
			{Type: llm.ContentText, Text: "line2"},
		},
	}
	if got := part.ToolResultText(); got != "line1\nline2" {
		t.Errorf("ToolResultText() = %q, want %q (image must be dropped)", got, "line1\nline2")
	}

	// Plain ToolResultContent takes precedence / works as before.
	plain := llm.ContentPart{Type: llm.ContentToolResult, ToolResultContent: "plain"}
	if got := plain.ToolResultText(); got != "plain" {
		t.Errorf("ToolResultText() plain = %q", got)
	}

	// Image-only result degrades to empty string (no panic).
	imgOnly := llm.ContentPart{Type: llm.ContentToolResult, ToolResultParts: []llm.ContentPart{
		{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}},
	}}
	if got := imgOnly.ToolResultText(); got != "" {
		t.Errorf("image-only ToolResultText() = %q, want empty", got)
	}
}

func TestNewToolResultParts(t *testing.T) {
	msg := llm.NewToolResultParts("call_9", true,
		llm.ContentPart{Type: llm.ContentText, Text: "err"},
	)
	if msg.Role != llm.RoleUser || len(msg.Content) != 1 {
		t.Fatalf("unexpected message: %+v", msg)
	}
	p := msg.Content[0]
	if p.Type != llm.ContentToolResult || p.ToolResultID != "call_9" || !p.IsError {
		t.Errorf("tool result part wrong: %+v", p)
	}
	if len(p.ToolResultParts) != 1 || p.ToolResultParts[0].Text != "err" {
		t.Errorf("parts not set: %+v", p.ToolResultParts)
	}
}
