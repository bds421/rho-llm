package llm

import (
	"context"
	"fmt"
	"strings"
)

// ModalityClient is the provider-neutral synchronous interface for non-chat
// model operations. Wire protocols register their implementation through
// RegisterModalityDriver; callers construct clients with NewModalityClient and
// reuse them so the underlying safe HTTP transport remains pooled.
type ModalityClient interface {
	GenerateEmbeddings(context.Context, EmbeddingRequest) (*EmbeddingResponse, error)
	GenerateImages(context.Context, ImageRequest) (*ImageResponse, error)
	SynthesizeSpeech(context.Context, SpeechRequest) (*SpeechResponse, error)
	TranscribeAudio(context.Context, TranscriptionRequest) (string, error)
	Provider() string
	Model() string
	Close() error
}

// ModalityDriver is the protocol adapter contract registered by provider
// packages. Validate* methods must be pure: they decide only whether the
// protocol can encode an otherwise provider-neutral request. New constructs the
// durable network client used for dispatch.
type ModalityDriver interface {
	New(Config) (ModalityClient, error)
	ValidateEmbeddingRequest(Config, EmbeddingRequest) error
	ValidateImageRequest(Config, ImageRequest) error
	ValidateSpeechRequest(Config, SpeechRequest) error
	ValidateTranscriptionRequest(Config, TranscriptionRequest) error
}

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

// ImageRequest requests generated images from a prompt. Exact geometry, count,
// and format are application-owned parameters; rho only validates whether the
// selected adapter can encode them.
type ImageRequest struct {
	Model        string
	Prompt       string
	N            int
	WidthPixels  uint32
	HeightPixels uint32
	MediaType    string
}

// GeneratedImage carries provider-returned bytes and their verified media type.
// OpenAI-compatible adapters require B64JSON and reject URL-only results because
// an un-fetched URL cannot prove artifact identity or content type.
type GeneratedImage struct {
	MediaType string
	B64JSON   string
	URL       string
}

// ImageResponse holds the generated images.
type ImageResponse struct {
	Images []GeneratedImage
}

// SpeechRequest synthesizes speech from text. Voice and format are selected by
// the application rather than inferred by the intelligence transport.
type SpeechRequest struct {
	Model     string
	Input     string
	Voice     string
	MediaType string
}

// SpeechResponse carries synthesized bytes and their verified media type.
type SpeechResponse struct {
	Audio     []byte
	MediaType string
}

// TranscriptionRequest transcribes audio bytes to text.
type TranscriptionRequest struct {
	Model     string
	Audio     []byte
	MediaType string
	Language  string
}

// ValidateEmbeddingRequest proves that req is supported by reviewed capability
// metadata and encodable by the registered protocol adapter. It performs no
// network I/O.
func ValidateEmbeddingRequest(cfg Config, req EmbeddingRequest) error {
	if len(req.Input) == 0 {
		return fmt.Errorf("llm: embeddings require at least one input")
	}
	if err := RequireCapabilitiesForModel(cfg, req.Model, CapabilityEmbeddings); err != nil {
		return err
	}
	driver, err := modalityDriverFor(cfg)
	if err != nil {
		return err
	}
	return driver.ValidateEmbeddingRequest(cfg, req)
}

// ValidateImageRequest proves that req is supported by reviewed capability
// metadata and encodable by the registered protocol adapter. It deliberately
// imposes no global image-size or count ceiling.
func ValidateImageRequest(cfg Config, req ImageRequest) error {
	if strings.TrimSpace(req.Prompt) == "" {
		return fmt.Errorf("llm: image generation requires a prompt")
	}
	if req.N < 0 {
		return fmt.Errorf("llm: image count must be non-negative")
	}
	if (req.WidthPixels == 0) != (req.HeightPixels == 0) {
		return fmt.Errorf("llm: image output geometry must include width and height")
	}
	if err := RequireCapabilitiesForModel(cfg, req.Model, CapabilityImageGeneration); err != nil {
		return err
	}
	driver, err := modalityDriverFor(cfg)
	if err != nil {
		return err
	}
	return driver.ValidateImageRequest(cfg, req)
}

// ValidateSpeechRequest proves that req is supported by reviewed capability
// metadata and encodable by the registered protocol adapter.
func ValidateSpeechRequest(cfg Config, req SpeechRequest) error {
	if strings.TrimSpace(req.Input) == "" {
		return fmt.Errorf("llm: speech synthesis requires input text")
	}
	if strings.TrimSpace(req.Voice) == "" || req.Voice != strings.TrimSpace(req.Voice) {
		return fmt.Errorf("llm: speech synthesis requires an exact non-empty voice")
	}
	if err := RequireCapabilitiesForModel(cfg, req.Model, CapabilitySpeechSynthesis); err != nil {
		return err
	}
	driver, err := modalityDriverFor(cfg)
	if err != nil {
		return err
	}
	return driver.ValidateSpeechRequest(cfg, req)
}

// ValidateTranscriptionRequest proves that req is supported by reviewed
// capability metadata and encodable by the registered protocol adapter.
func ValidateTranscriptionRequest(cfg Config, req TranscriptionRequest) error {
	if len(req.Audio) == 0 {
		return fmt.Errorf("llm: transcription requires audio data")
	}
	if err := RequireCapabilitiesForModel(cfg, req.Model, CapabilityTranscription); err != nil {
		return err
	}
	driver, err := modalityDriverFor(cfg)
	if err != nil {
		return err
	}
	return driver.ValidateTranscriptionRequest(cfg, req)
}

func modalityDriverFor(cfg Config) (ModalityDriver, error) {
	protocol := ResolveProtocol(cfg)
	driver := getModalityDriver(protocol)
	if driver == nil {
		return nil, fmt.Errorf(
			"llm: no registered modality driver for protocol %q (provider %q)",
			protocol, cfg.Provider,
		)
	}
	return driver, nil
}
