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
	_ "github.com/bds421/rho-llm/provider/gemini"
)

// minimal 1x1 PNG
var tinyPNG = base64.StdEncoding.EncodeToString([]byte{
	0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a,
	0x00, 0x00, 0x00, 0x0d, 'I', 'H', 'D', 'R',
	0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
	0xde, 0x00, 0x00, 0x00, 0x0c, 'I', 'D', 'A', 'T',
	0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00, 0x00,
	0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0xfe, 0xd4,
	0xef, 0x00, 0x00, 0x00, 0x00, 'I', 'E', 'N', 'D',
	0xae, 0x42, 0x60, 0x82,
})

func TestGeminiModalityEmbeddingsAndImages(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(r.URL.Path, "embedContent") {
			_, _ = w.Write([]byte(`{"embedding":{"values":[0.25,0.5,0.75]}}`))
			return
		}
		if strings.Contains(r.URL.Path, "generateContent") {
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"` + tinyPNG + `"}}]}}]}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	// Register reviewed modality models for this test process.
	_ = llm.RegisterModel(llm.ModelInfo{
		ID: "gemini-embedding-001", Provider: "gemini",
		Capabilities: llm.Capabilities(llm.CapabilityEmbeddings),
	})
	_ = llm.RegisterModel(llm.ModelInfo{
		ID: "gemini-2.5-flash-image", Provider: "gemini",
		Capabilities: llm.Capabilities(llm.CapabilityImageGeneration),
	})

	embClient, err := llm.NewModalityClient(llm.Config{
		Provider: "gemini", Model: "gemini-embedding-001", APIKey: "k",
		BaseURL: srv.URL + "/v1beta/models", DisableProxy: true, DisableRetries: true,
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewModalityClient embeddings: %v", err)
	}
	defer embClient.Close()
	emb, err := embClient.GenerateEmbeddings(context.Background(), llm.EmbeddingRequest{
		Model: "gemini-embedding-001", Input: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("GenerateEmbeddings: %v", err)
	}
	if len(emb.Embeddings) != 1 || len(emb.Embeddings[0].Vector) != 3 {
		t.Fatalf("emb=%+v", emb)
	}

	imgClient, err := llm.NewModalityClient(llm.Config{
		Provider: "gemini", Model: "gemini-2.5-flash-image", APIKey: "k",
		BaseURL: srv.URL + "/v1beta/models", DisableProxy: true, DisableRetries: true,
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewModalityClient images: %v", err)
	}
	defer imgClient.Close()
	img, err := imgClient.GenerateImages(context.Background(), llm.ImageRequest{
		Model: "gemini-2.5-flash-image", Prompt: "a red square", N: 1, MediaType: "image/png",
	})
	if err != nil {
		t.Fatalf("GenerateImages: %v", err)
	}
	if len(img.Images) != 1 || img.Images[0].B64JSON == "" {
		t.Fatalf("img=%+v", img)
	}
}

func TestGeminiModalityRejectsSpeech(t *testing.T) {
	// Speech is outside the Gemini protocol envelope — fail closed before transport.
	err := llm.ValidateSpeechRequest(llm.Config{
		Provider: "gemini", Model: "gemini-2.5-flash",
		ModelCapabilities: llm.Capabilities(llm.CapabilitySpeechSynthesis),
	}, llm.SpeechRequest{Model: "gemini-2.5-flash", Input: "hi", Voice: "x", MediaType: "audio/mpeg"})
	if err == nil || !strings.Contains(err.Error(), "speech") {
		t.Fatalf("want speech rejected, got %v", err)
	}
}
