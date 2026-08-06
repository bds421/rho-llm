package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/bds421/rho-llm"
)

func init() {
	// Chat provider already registers in the other init; modality driver is additive.
	llm.RegisterModalityDriver("gemini", modalityDriver{})
}

type modalityDriver struct{}

func (modalityDriver) New(cfg llm.Config) (llm.ModalityClient, error) {
	return New(cfg)
}

func (modalityDriver) ValidateEmbeddingRequest(llm.Config, llm.EmbeddingRequest) error {
	return nil
}

func (modalityDriver) ValidateImageRequest(_ llm.Config, req llm.ImageRequest) error {
	if req.MediaType != "" && req.MediaType != "image/png" && req.MediaType != "image/jpeg" && req.MediaType != "image/webp" {
		return fmt.Errorf("gemini: unsupported image output media type %q", req.MediaType)
	}
	return nil
}

func (modalityDriver) ValidateSpeechRequest(llm.Config, llm.SpeechRequest) error {
	return fmt.Errorf("gemini: speech synthesis is not supported")
}

func (modalityDriver) ValidateTranscriptionRequest(llm.Config, llm.TranscriptionRequest) error {
	return fmt.Errorf("gemini: transcription is not supported")
}

// GenerateEmbeddings calls embedContent once per input string.
func (c *Client) GenerateEmbeddings(ctx context.Context, req llm.EmbeddingRequest) (*llm.EmbeddingResponse, error) {
	model := req.Model
	if model == "" {
		model = c.config.Model
	}
	out := &llm.EmbeddingResponse{Model: model, Embeddings: make([]llm.Embedding, 0, len(req.Input))}
	var inputTokens int
	for i, text := range req.Input {
		endpoint := fmt.Sprintf("%s/%s:embedContent", c.baseURL, url.PathEscape(model))
		body, err := json.Marshal(map[string]any{
			"content": map[string]any{
				"parts": []map[string]any{{"text": text}},
			},
		})
		if err != nil {
			return nil, err
		}
		resp, err := c.doModalityJSON(ctx, endpoint, body)
		if err != nil {
			return nil, err
		}
		var wire struct {
			Embedding struct {
				Values []float64 `json:"values"`
			} `json:"embedding"`
			// Some revisions nest under embedding.values only.
		}
		if err := llm.DecodeJSONResponse(resp, c.config, &wire); err != nil {
			_ = resp.Body.Close()
			return nil, fmt.Errorf("gemini: decode embeddings: %w", err)
		}
		_ = resp.Body.Close()
		if len(wire.Embedding.Values) == 0 {
			return nil, fmt.Errorf("gemini: empty embedding vector")
		}
		out.Embeddings = append(out.Embeddings, llm.Embedding{Index: i, Vector: wire.Embedding.Values})
		inputTokens += len(text) / 4 // best-effort; usage often omitted
	}
	out.InputTokens = inputTokens
	return out, nil
}

// GenerateImages uses generateContent with IMAGE response modality.
func (c *Client) GenerateImages(ctx context.Context, req llm.ImageRequest) (*llm.ImageResponse, error) {
	model := req.Model
	if model == "" {
		model = c.config.Model
	}
	n := req.N
	if n <= 0 {
		n = 1
	}
	images := make([]llm.GeneratedImage, 0, n)
	for i := 0; i < n; i++ {
		endpoint := fmt.Sprintf("%s/%s:generateContent", c.baseURL, url.PathEscape(model))
		body, err := json.Marshal(map[string]any{
			"contents": []map[string]any{{
				"role":  "user",
				"parts": []map[string]any{{"text": req.Prompt}},
			}},
			"generationConfig": map[string]any{
				"responseModalities": []string{"TEXT", "IMAGE"},
			},
		})
		if err != nil {
			return nil, err
		}
		resp, err := c.doModalityJSON(ctx, endpoint, body)
		if err != nil {
			return nil, err
		}
		var wire geminiResponse
		if err := llm.DecodeJSONResponse(resp, c.config, &wire); err != nil {
			_ = resp.Body.Close()
			return nil, fmt.Errorf("gemini: decode image response: %w", err)
		}
		_ = resp.Body.Close()
		found := false
		for _, cand := range wire.Candidates {
			for _, part := range cand.Content.Parts {
				if part.InlineData == nil || part.InlineData.Data == "" {
					continue
				}
				mediaType := part.InlineData.MimeType
				if mediaType == "" {
					mediaType = "image/png"
				}
				if err := verifyImageB64(part.InlineData.Data, mediaType); err != nil {
					return nil, err
				}
				images = append(images, llm.GeneratedImage{
					MediaType: mediaType,
					B64JSON:   part.InlineData.Data,
				})
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("gemini: image response contained no inline image bytes")
		}
	}
	return &llm.ImageResponse{Images: images}, nil
}

// SynthesizeSpeech is unsupported on Gemini.
func (c *Client) SynthesizeSpeech(context.Context, llm.SpeechRequest) (*llm.SpeechResponse, error) {
	return nil, fmt.Errorf("gemini: speech synthesis is not supported")
}

// TranscribeAudio is unsupported on Gemini as a dedicated modality.
func (c *Client) TranscribeAudio(context.Context, llm.TranscriptionRequest) (string, error) {
	return "", fmt.Errorf("gemini: transcription is not supported")
}

func (c *Client) doModalityJSON(ctx context.Context, endpoint string, body []byte) (*http.Response, error) {
	resp, err := llm.DoHTTP(ctx, c.config, c.httpClient, func(ctx context.Context) (*http.Request, error) {
		req, err := llm.NewJSONRequest(ctx, endpoint, body)
		if err != nil {
			return nil, err
		}
		req.Header.Set("x-goog-api-key", c.config.APIKey)
		return req, nil
	})
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		err := llm.ErrorFromResponse("gemini", resp, c.config)
		_ = resp.Body.Close()
		return nil, err
	}
	return resp, nil
}

func verifyImageB64(b64, mediaType string) error {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		// try raw std without padding issues
		raw, err = base64.RawStdEncoding.DecodeString(strings.TrimRight(b64, "="))
		if err != nil {
			return fmt.Errorf("gemini: image bytes are not valid base64")
		}
	}
	if len(raw) < 8 {
		return fmt.Errorf("gemini: image payload too short")
	}
	switch mediaType {
	case "image/png":
		if !(raw[0] == 0x89 && raw[1] == 'P' && raw[2] == 'N' && raw[3] == 'G') {
			return fmt.Errorf("gemini: payload is not PNG")
		}
	case "image/jpeg":
		if !(raw[0] == 0xff && raw[1] == 0xd8) {
			return fmt.Errorf("gemini: payload is not JPEG")
		}
	case "image/webp":
		if string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WEBP" {
			return fmt.Errorf("gemini: payload is not WebP")
		}
	}
	return nil
}
