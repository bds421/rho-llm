package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
)

// This file adds non-chat capabilities for OpenAI-compatible providers — embeddings,
// image generation, and audio (speech/transcription) — as standalone functions that
// take a Config (provider/BaseURL/APIKey/AuthHeader). They reuse SafeHTTPClient
// (TLS 1.2+, redirect header stripping) and honor BlockPrivateBaseURL. Providers
// with non-OpenAI shapes (e.g. Gemini Imagen, Anthropic) are out of scope here.

// =============================================================================
// Embeddings — POST {BaseURL}/embeddings
// =============================================================================

// EmbeddingRequest requests vector embeddings for one or more inputs.
type EmbeddingRequest struct {
	Model string
	Input []string
}

// Embedding is one input's vector, with its position in the request.
type Embedding struct {
	Index  int
	Vector []float64
}

// EmbeddingResponse holds the embeddings and token usage.
type EmbeddingResponse struct {
	Model       string
	Embeddings  []Embedding
	InputTokens int
}

// GenerateEmbeddings calls an OpenAI-compatible /embeddings endpoint.
func GenerateEmbeddings(ctx context.Context, cfg Config, req EmbeddingRequest) (*EmbeddingResponse, error) {
	if len(req.Input) == 0 {
		return nil, fmt.Errorf("llm: embeddings require at least one input")
	}
	var out struct {
		Model string `json:"model"`
		Data  []struct {
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := postJSON(ctx, cfg, "/embeddings", map[string]any{"model": req.Model, "input": req.Input}, &out); err != nil {
		return nil, err
	}
	res := &EmbeddingResponse{Model: out.Model, InputTokens: out.Usage.PromptTokens}
	for _, d := range out.Data {
		res.Embeddings = append(res.Embeddings, Embedding{Index: d.Index, Vector: d.Embedding})
	}
	return res, nil
}

// =============================================================================
// Image generation — POST {BaseURL}/images/generations
// =============================================================================

// ImageRequest requests generated images from a prompt.
type ImageRequest struct {
	Model  string
	Prompt string
	N      int    // number of images (0 = provider default)
	Size   string // e.g. "1024x1024" (empty = provider default)
}

// GeneratedImage holds one result, as base64 (b64_json) and/or a URL.
type GeneratedImage struct {
	B64JSON string
	URL     string
}

// ImageResponse holds the generated images.
type ImageResponse struct {
	Images []GeneratedImage
}

// GenerateImages calls an OpenAI-compatible /images/generations endpoint and
// requests base64 (b64_json) output.
func GenerateImages(ctx context.Context, cfg Config, req ImageRequest) (*ImageResponse, error) {
	if req.Prompt == "" {
		return nil, fmt.Errorf("llm: image generation requires a prompt")
	}
	body := map[string]any{"model": req.Model, "prompt": req.Prompt, "response_format": "b64_json"}
	if req.N > 0 {
		body["n"] = req.N
	}
	if req.Size != "" {
		body["size"] = req.Size
	}
	var out struct {
		Data []struct {
			B64JSON string `json:"b64_json"`
			URL     string `json:"url"`
		} `json:"data"`
	}
	if err := postJSON(ctx, cfg, "/images/generations", body, &out); err != nil {
		return nil, err
	}
	res := &ImageResponse{}
	for _, d := range out.Data {
		res.Images = append(res.Images, GeneratedImage{B64JSON: d.B64JSON, URL: d.URL})
	}
	return res, nil
}

// =============================================================================
// Audio out — POST {BaseURL}/audio/speech
// =============================================================================

// SpeechRequest synthesizes speech from text.
type SpeechRequest struct {
	Model  string
	Input  string
	Voice  string
	Format string // mp3, wav, opus, ... (empty = provider default)
}

// SynthesizeSpeech returns the raw audio bytes for the request.
func SynthesizeSpeech(ctx context.Context, cfg Config, req SpeechRequest) ([]byte, error) {
	if req.Input == "" {
		return nil, fmt.Errorf("llm: speech synthesis requires input text")
	}
	body := map[string]any{"model": req.Model, "input": req.Input, "voice": req.Voice}
	if req.Format != "" {
		body["response_format"] = req.Format
	}
	return doPost(ctx, cfg, "/audio/speech", body)
}

// =============================================================================
// Audio in — POST {BaseURL}/audio/transcriptions (multipart)
// =============================================================================

// TranscriptionRequest transcribes audio bytes to text.
type TranscriptionRequest struct {
	Model    string
	Audio    []byte
	Filename string // e.g. "audio.mp3" (used for the multipart part)
	Language string // optional ISO-639-1 hint
}

// TranscribeAudio uploads audio to an OpenAI-compatible /audio/transcriptions
// endpoint (multipart/form-data) and returns the transcript text.
func TranscribeAudio(ctx context.Context, cfg Config, req TranscriptionRequest) (string, error) {
	if len(req.Audio) == 0 {
		return "", fmt.Errorf("llm: transcription requires audio data")
	}
	if err := CheckBaseURL(cfg); err != nil {
		return "", err
	}
	base := ResolveBaseURL(cfg)
	if base == "" {
		return "", fmt.Errorf("llm: no base URL for provider %q", cfg.Provider)
	}
	filename := req.Filename
	if filename == "" {
		filename = "audio"
	}

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	if err := mw.WriteField("model", req.Model); err != nil {
		return "", err
	}
	if req.Language != "" {
		_ = mw.WriteField("language", req.Language)
	}
	fw, err := mw.CreateFormFile("file", filename)
	if err != nil {
		return "", err
	}
	if _, err := fw.Write(req.Audio); err != nil {
		return "", err
	}
	if err := mw.Close(); err != nil {
		return "", err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", base+"/audio/transcriptions", &buf)
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("Content-Type", mw.FormDataContentType())
	setAuth(httpReq, cfg)

	data, err := doRaw(ctx, cfg, httpReq)
	if err != nil {
		return "", err
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return "", fmt.Errorf("llm: decode transcription response: %w", err)
	}
	return out.Text, nil
}

// =============================================================================
// shared HTTP plumbing
// =============================================================================

func postJSON(ctx context.Context, cfg Config, path string, body any, out any) error {
	data, err := doPost(ctx, cfg, path, body)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("llm: decode %s response: %w", path, err)
	}
	return nil
}

func doPost(ctx context.Context, cfg Config, path string, body any) ([]byte, error) {
	if err := CheckBaseURL(cfg); err != nil {
		return nil, err
	}
	base := ResolveBaseURL(cfg)
	if base == "" {
		return nil, fmt.Errorf("llm: no base URL for provider %q", cfg.Provider)
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", base+path, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	setAuth(httpReq, cfg)
	return doRaw(ctx, cfg, httpReq)
}

func doRaw(_ context.Context, cfg Config, httpReq *http.Request) ([]byte, error) {
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	resp, err := SafeHTTPClient(timeout).Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, cfg.EffectiveMaxResponseBodyBytes()))
	if resp.StatusCode != http.StatusOK {
		name := cfg.ProviderName
		if name == "" {
			name = cfg.Provider
		}
		return nil, NewAPIErrorFromStatusWithLimit(name, resp.StatusCode, string(data), cfg.EffectiveMaxErrorMessageLen())
	}
	return data, nil
}

func setAuth(httpReq *http.Request, cfg Config) {
	authHeader := ResolveAuthHeader(cfg)
	if authHeader != "" && cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", authHeader+" "+cfg.APIKey)
	}
}
