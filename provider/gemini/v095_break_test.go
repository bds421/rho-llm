package gemini_test

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

// H4: a 200 image response whose inline payload is 8–11 bytes and labelled
// image/webp used to index raw[8:12] past the end and panic the consumer's
// process. Every length in the window must return an error, never panic.
func TestGeminiImageWebPShortPayloadDoesNotPanic(t *testing.T) {
	_ = llm.RegisterModel(llm.ModelInfo{
		ID: "gemini-2.5-flash-image", Provider: "gemini",
		Capabilities: llm.Capabilities(llm.CapabilityImageGeneration),
	})
	for n := 8; n <= 11; n++ {
		raw := []byte("RIFF\x00\x00\x00\x00WEB")[:n]
		payload := base64.StdEncoding.EncodeToString(raw)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/webp","data":"` + payload + `"}}]}}]}`))
		}))
		client, err := llm.NewModalityClient(llm.Config{
			Provider: "gemini", Model: "gemini-2.5-flash-image", APIKey: "k",
			BaseURL: srv.URL + "/v1beta/models", DisableProxy: true, DisableRetries: true,
			Timeout: 5 * time.Second,
		})
		if err != nil {
			srv.Close()
			t.Fatalf("NewModalityClient: %v", err)
		}
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("len=%d: GenerateImages panicked: %v", n, r)
				}
			}()
			_, err = client.GenerateImages(context.Background(), llm.ImageRequest{
				Model: "gemini-2.5-flash-image", Prompt: "x", N: 1, MediaType: "image/webp",
			})
			if err == nil || !strings.Contains(err.Error(), "WebP") {
				t.Errorf("len=%d: want a not-WebP error, got %v", n, err)
			}
		}()
		_ = client.Close()
		srv.Close()
	}
}
