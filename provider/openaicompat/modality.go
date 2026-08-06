package openaicompat

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"mime"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"

	llm "github.com/bds421/rho-llm"
)

// modalityDriver keeps OpenAI-compatible endpoint and format knowledge inside
// the protocol adapter. Its validation methods are pure and are used by rho's
// public fail-before-dispatch validators.
type modalityDriver struct{}

func (modalityDriver) New(cfg llm.Config) (llm.ModalityClient, error) { return New(cfg) }

func (modalityDriver) ValidateEmbeddingRequest(llm.Config, llm.EmbeddingRequest) error {
	return nil
}

func (modalityDriver) ValidateImageRequest(_ llm.Config, req llm.ImageRequest) error {
	_, err := imageOutputFormat(req.MediaType)
	return err
}

func (modalityDriver) ValidateSpeechRequest(_ llm.Config, req llm.SpeechRequest) error {
	_, err := speechOutputFormat(req.MediaType)
	return err
}

func (modalityDriver) ValidateTranscriptionRequest(_ llm.Config, req llm.TranscriptionRequest) error {
	if _, err := transcriptionExtension(req.MediaType); err != nil {
		return err
	}
	if err := validateTranscriptionLanguage(req.Language); err != nil {
		return err
	}
	if actual := transcriptionMediaTypeFromSignature(req.Audio); actual == "" {
		return fmt.Errorf("openaicompat: transcription audio has no supported signature")
	} else if !sameTranscriptionMediaType(actual, req.MediaType) {
		return fmt.Errorf(
			"openaicompat: transcription audio media type %q does not match declared %q",
			actual, req.MediaType,
		)
	}
	return nil
}

type embeddingsAPIResponse struct {
	Model string `json:"model"`
	Data  []struct {
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
	} `json:"usage"`
}

func embeddingsRequestBody(req llm.EmbeddingRequest) map[string]any {
	return map[string]any{"model": req.Model, "input": req.Input}
}

func (wire *embeddingsAPIResponse) toResponse(expectedInputs int) (*llm.EmbeddingResponse, error) {
	if len(wire.Data) == 0 {
		return nil, fmt.Errorf("openaicompat: embeddings response contains no vectors")
	}
	if expectedInputs > 0 && len(wire.Data) != expectedInputs {
		return nil, fmt.Errorf(
			"openaicompat: embeddings response contains %d vectors for %d inputs",
			len(wire.Data), expectedInputs,
		)
	}
	if wire.Usage.PromptTokens < 0 {
		return nil, fmt.Errorf("openaicompat: embeddings response has negative input-token usage")
	}
	seen := make(map[int]struct{}, len(wire.Data))
	result := &llm.EmbeddingResponse{
		Model: wire.Model, InputTokens: wire.Usage.PromptTokens,
		Embeddings: make([]llm.Embedding, 0, len(wire.Data)),
	}
	for _, item := range wire.Data {
		if item.Index < 0 || (expectedInputs > 0 && item.Index >= expectedInputs) {
			return nil, fmt.Errorf("openaicompat: embeddings response has invalid index %d", item.Index)
		}
		if _, duplicate := seen[item.Index]; duplicate {
			return nil, fmt.Errorf("openaicompat: embeddings response has duplicate index %d", item.Index)
		}
		if len(item.Embedding) == 0 {
			return nil, fmt.Errorf("openaicompat: embedding %d is empty", item.Index)
		}
		for _, value := range item.Embedding {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("openaicompat: embedding %d contains a non-finite value", item.Index)
			}
		}
		seen[item.Index] = struct{}{}
		result.Embeddings = append(result.Embeddings, llm.Embedding{
			Index: item.Index, Vector: item.Embedding,
		})
	}
	return result, nil
}

// GenerateEmbeddings calls the OpenAI-compatible /embeddings endpoint.
func (c *Client) GenerateEmbeddings(
	ctx context.Context, req llm.EmbeddingRequest,
) (*llm.EmbeddingResponse, error) {
	model := c.modalityModel(req.Model)
	req.Model = model
	body, err := json.Marshal(embeddingsRequestBody(req))
	if err != nil {
		return nil, fmt.Errorf("openaicompat: encode embeddings request: %w", err)
	}
	response, err := c.doModalityRequest(ctx, func(ctx context.Context) (*http.Request, error) {
		return c.newJSONModalityRequest(ctx, "/embeddings", body)
	})
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	var wire embeddingsAPIResponse
	if err := decodeBoundedJSON(response.Body, c.config.EffectiveMaxResponseBodyBytes(), &wire); err != nil {
		return nil, fmt.Errorf("openaicompat: decode embeddings response: %w", err)
	}
	return wire.toResponse(len(req.Input))
}

type imageAPIResponse struct {
	Data []struct {
		B64JSON string `json:"b64_json"`
		URL     string `json:"url"`
	} `json:"data"`
}

// GenerateImages calls /images/generations and requires inline base64 output so
// the adapter can verify the artifact bytes before returning them.
func (c *Client) GenerateImages(
	ctx context.Context, req llm.ImageRequest,
) (*llm.ImageResponse, error) {
	format, _ := imageOutputFormat(req.MediaType)
	bodyValue := map[string]any{
		"model": c.modalityModel(req.Model), "prompt": req.Prompt,
		"response_format": "b64_json",
	}
	if req.N > 0 {
		bodyValue["n"] = req.N
	}
	if req.WidthPixels != 0 {
		bodyValue["size"] = strconv.FormatUint(uint64(req.WidthPixels), 10) + "x" +
			strconv.FormatUint(uint64(req.HeightPixels), 10)
	}
	if format != "" {
		bodyValue["output_format"] = format
	}
	body, err := json.Marshal(bodyValue)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: encode image request: %w", err)
	}
	response, err := c.doModalityRequest(ctx, func(ctx context.Context) (*http.Request, error) {
		return c.newJSONModalityRequest(ctx, "/images/generations", body)
	})
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	var wire imageAPIResponse
	if err := decodeBoundedJSON(response.Body, c.config.EffectiveMaxResponseBodyBytes(), &wire); err != nil {
		return nil, fmt.Errorf("openaicompat: decode image response: %w", err)
	}
	if len(wire.Data) == 0 {
		return nil, fmt.Errorf("openaicompat: image response contains no artifacts")
	}
	if req.N > 0 && len(wire.Data) != req.N {
		return nil, fmt.Errorf(
			"openaicompat: image response contains %d artifacts; requested %d",
			len(wire.Data), req.N,
		)
	}
	result := &llm.ImageResponse{Images: make([]llm.GeneratedImage, 0, len(wire.Data))}
	for index, image := range wire.Data {
		mediaType, err := verifyGeneratedImage(req.MediaType, image.B64JSON, image.URL)
		if err != nil {
			return nil, fmt.Errorf("openaicompat: generated image %d: %w", index, err)
		}
		result.Images = append(result.Images, llm.GeneratedImage{
			MediaType: mediaType, B64JSON: image.B64JSON, URL: image.URL,
		})
	}
	return result, nil
}

// SynthesizeSpeech calls /audio/speech and verifies Content-Type against the
// actual bounded payload signature.
func (c *Client) SynthesizeSpeech(
	ctx context.Context, req llm.SpeechRequest,
) (*llm.SpeechResponse, error) {
	format, _ := speechOutputFormat(req.MediaType)
	bodyValue := map[string]any{
		"model": c.modalityModel(req.Model), "input": req.Input, "voice": req.Voice,
	}
	if format != "" {
		bodyValue["response_format"] = format
	}
	body, err := json.Marshal(bodyValue)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: encode speech request: %w", err)
	}
	response, err := c.doModalityRequest(ctx, func(ctx context.Context) (*http.Request, error) {
		return c.newJSONModalityRequest(ctx, "/audio/speech", body)
	})
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	data, err := readBoundedBody(response.Body, c.config.EffectiveMaxResponseBodyBytes())
	if err != nil {
		return nil, fmt.Errorf("openaicompat: incomplete speech response: %w", err)
	}
	mediaType, err := verifySpeechMediaType(req.MediaType, response.Header.Get("Content-Type"), data)
	if err != nil {
		return nil, err
	}
	return &llm.SpeechResponse{Audio: data, MediaType: mediaType}, nil
}

// TranscribeAudio uploads a protocol-specific multipart request to
// /audio/transcriptions.
func (c *Client) TranscribeAudio(ctx context.Context, req llm.TranscriptionRequest) (string, error) {
	extension, _ := transcriptionExtension(req.MediaType)
	var payload bytes.Buffer
	multipartWriter := multipart.NewWriter(&payload)
	if err := multipartWriter.WriteField("model", c.modalityModel(req.Model)); err != nil {
		return "", err
	}
	if req.Language != "" {
		if err := multipartWriter.WriteField("language", req.Language); err != nil {
			return "", err
		}
	}
	fileWriter, err := multipartWriter.CreateFormFile("file", "audio."+extension)
	if err != nil {
		return "", err
	}
	if _, err := fileWriter.Write(req.Audio); err != nil {
		return "", err
	}
	if err := multipartWriter.Close(); err != nil {
		return "", err
	}
	data := append([]byte(nil), payload.Bytes()...)
	contentType := multipartWriter.FormDataContentType()
	response, err := c.doModalityRequest(ctx, func(ctx context.Context) (*http.Request, error) {
		httpRequest, buildErr := http.NewRequestWithContext(
			ctx, http.MethodPost, c.baseURL+"/audio/transcriptions", bytes.NewReader(data),
		)
		if buildErr != nil {
			return nil, buildErr
		}
		httpRequest.Header.Set("Content-Type", contentType)
		c.setModalityAuth(httpRequest)
		return httpRequest, nil
	})
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	var wire struct {
		Text string `json:"text"`
	}
	if err := decodeBoundedJSON(response.Body, c.config.EffectiveMaxResponseBodyBytes(), &wire); err != nil {
		return "", fmt.Errorf("openaicompat: decode transcription response: %w", err)
	}
	return wire.Text, nil
}

func (c *Client) modalityModel(requested string) string {
	if strings.TrimSpace(requested) != "" {
		return llm.ResolveModelAlias(strings.TrimSpace(requested))
	}
	return c.config.Model
}

func (c *Client) newJSONModalityRequest(
	ctx context.Context, path string, body []byte,
) (*http.Request, error) {
	request, err := llm.NewJSONRequest(ctx, c.baseURL+path, body)
	if err != nil {
		return nil, err
	}
	c.setModalityAuth(request)
	return request, nil
}

func (c *Client) setModalityAuth(request *http.Request) {
	if c.authHeader != "" && c.config.APIKey != "" {
		request.Header.Set("Authorization", c.authHeader+" "+c.config.APIKey)
	}
}

func (c *Client) doModalityRequest(
	ctx context.Context, build llm.HTTPRequestFactory,
) (*http.Response, error) {
	response, err := llm.DoHTTP(ctx, c.config, c.httpClient, build)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: request failed: %w", err)
	}
	if response.StatusCode/100 != 2 {
		defer response.Body.Close()
		return nil, llm.ErrorFromResponse(c.providerName, response, c.config)
	}
	return response, nil
}

func imageOutputFormat(mediaType string) (string, error) {
	switch mediaType {
	case "":
		return "", nil
	case "image/png":
		return "png", nil
	case "image/jpeg":
		return "jpeg", nil
	case "image/webp":
		return "webp", nil
	default:
		return "", fmt.Errorf("openaicompat: unsupported image output media type %q", mediaType)
	}
}

func verifyGeneratedImage(requestedMediaType, encoded, artifactURL string) (string, error) {
	if encoded == "" {
		if artifactURL != "" {
			return "", fmt.Errorf("provider returned URL-only image after b64_json was required")
		}
		return "", fmt.Errorf("provider returned neither image bytes nor URL")
	}
	decoded, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil {
		return "", fmt.Errorf("invalid base64 payload: %w", err)
	}
	if base64.StdEncoding.EncodeToString(decoded) != encoded {
		return "", fmt.Errorf("image payload is not canonical RFC 4648 base64")
	}
	actual := detectImageMediaType(decoded)
	if actual == "" {
		return "", fmt.Errorf("image payload has no supported signature")
	}
	if requestedMediaType != "" && actual != requestedMediaType {
		return "", fmt.Errorf("image payload media type %q does not match requested %q", actual, requestedMediaType)
	}
	return actual, nil
}

func detectImageMediaType(data []byte) string {
	switch {
	case len(data) >= 8 && bytes.Equal(data[:8], []byte("\x89PNG\r\n\x1a\n")):
		return "image/png"
	case len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff:
		return "image/jpeg"
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP")):
		return "image/webp"
	default:
		return ""
	}
}

func speechOutputFormat(mediaType string) (string, error) {
	switch mediaType {
	case "":
		return "", nil
	case "audio/mpeg":
		return "mp3", nil
	case "audio/ogg; codecs=opus":
		return "opus", nil
	case "audio/aac":
		return "aac", nil
	case "audio/flac":
		return "flac", nil
	case "audio/wav":
		return "wav", nil
	case "audio/L16":
		return "pcm", nil
	default:
		return "", fmt.Errorf("openaicompat: unsupported speech output media type %q", mediaType)
	}
}

func canonicalSpeechMediaType(mediaType string) (string, error) {
	switch mediaType {
	case "":
		return "", nil
	case "audio/mpeg":
		return "audio/mpeg", nil
	case "audio/ogg; codecs=opus":
		return "audio/ogg; codecs=opus", nil
	case "audio/aac":
		return "audio/aac", nil
	case "audio/flac":
		return "audio/flac", nil
	case "audio/wav":
		return "audio/wav", nil
	case "audio/L16":
		return "audio/L16", nil
	default:
		return "", fmt.Errorf("openaicompat: unsupported speech output media type %q", mediaType)
	}
}

func verifySpeechMediaType(requested, contentType string, data []byte) (string, error) {
	if len(data) == 0 {
		return "", fmt.Errorf("openaicompat: speech provider returned an empty body")
	}
	expected, err := canonicalSpeechMediaType(requested)
	if err != nil {
		return "", err
	}
	headerType, generic, err := speechHeaderMediaType(contentType)
	if err != nil {
		return "", err
	}
	actual := detectSpeechMediaType(data)
	if actual == "" {
		if expected == "audio/L16" && !generic && headerType == "audio/L16" {
			return "audio/L16", nil
		}
		return "", fmt.Errorf("openaicompat: speech payload has no supported verifiable signature")
	}
	if !generic && headerType != actual {
		return "", fmt.Errorf("openaicompat: speech Content-Type %q does not match payload %q", headerType, actual)
	}
	if expected != "" && expected != actual {
		return "", fmt.Errorf("openaicompat: speech payload media type %q does not match requested %q", actual, expected)
	}
	return actual, nil
}

func speechHeaderMediaType(contentType string) (mediaType string, generic bool, err error) {
	if strings.TrimSpace(contentType) == "" {
		return "", true, nil
	}
	parsed, parameters, parseErr := mime.ParseMediaType(contentType)
	if parseErr != nil {
		return "", false, fmt.Errorf("openaicompat: speech response Content-Type is invalid: %w", parseErr)
	}
	parsed = strings.ToLower(parsed)
	switch parsed {
	case "application/octet-stream", "binary/octet-stream":
		return "", true, nil
	case "audio/ogg":
		if codec := strings.ToLower(parameters["codecs"]); codec != "" && codec != "opus" {
			return "", false, fmt.Errorf("openaicompat: unsupported speech response Content-Type %q", contentType)
		}
		return "audio/ogg; codecs=opus", false, nil
	case "audio/opus":
		return "audio/ogg; codecs=opus", false, nil
	case "audio/mpeg", "audio/aac", "audio/flac", "audio/wav":
		return parsed, false, nil
	case "audio/x-wav":
		return "audio/wav", false, nil
	case "audio/l16":
		return "audio/L16", false, nil
	default:
		return "", false, fmt.Errorf("openaicompat: unsupported speech response Content-Type %q", contentType)
	}
}

func detectSpeechMediaType(data []byte) string {
	prefix := data
	if len(prefix) > 512 {
		prefix = prefix[:512]
	}
	switch {
	case len(prefix) >= 12 && bytes.Equal(prefix[:4], []byte("RIFF")) && bytes.Equal(prefix[8:12], []byte("WAVE")):
		return "audio/wav"
	case len(prefix) >= 4 && bytes.Equal(prefix[:4], []byte("fLaC")):
		return "audio/flac"
	case len(prefix) >= 8 && bytes.Equal(prefix[:4], []byte("OggS")) && bytes.Contains(prefix, []byte("OpusHead")):
		return "audio/ogg; codecs=opus"
	case len(prefix) >= 4 && bytes.Equal(prefix[:4], []byte("ADIF")):
		return "audio/aac"
	case len(prefix) >= 2 && prefix[0] == 0xff && prefix[1]&0xf6 == 0xf0:
		return "audio/aac"
	case len(prefix) >= 3 && bytes.Equal(prefix[:3], []byte("ID3")):
		return "audio/mpeg"
	case len(prefix) >= 2 && prefix[0] == 0xff && prefix[1]&0xe0 == 0xe0:
		return "audio/mpeg"
	default:
		return ""
	}
}

func transcriptionExtension(mediaType string) (string, error) {
	switch mediaType {
	case "audio/flac":
		return "flac", nil
	case "audio/m4a", "audio/mp4":
		return "m4a", nil
	case "audio/mpeg":
		return "mp3", nil
	case "audio/ogg":
		return "ogg", nil
	case "audio/wav", "audio/x-wav":
		return "wav", nil
	case "audio/webm":
		return "webm", nil
	default:
		return "", fmt.Errorf("openaicompat: unsupported transcription input media type %q", mediaType)
	}
}

func validateTranscriptionLanguage(language string) error {
	if language == "" {
		return nil
	}
	if len(language) != 2 || language[0] < 'a' || language[0] > 'z' ||
		language[1] < 'a' || language[1] > 'z' {
		return fmt.Errorf(
			"openaicompat: transcription language %q is not a lowercase ISO-639-1 code",
			language,
		)
	}
	return nil
}

func transcriptionMediaTypeFromSignature(data []byte) string {
	switch {
	case len(data) >= 4 && bytes.Equal(data[:4], []byte("fLaC")):
		return "audio/flac"
	case len(data) >= 12 && bytes.Equal(data[4:8], []byte("ftyp")):
		return "audio/mp4"
	case len(data) >= 3 && bytes.Equal(data[:3], []byte("ID3")):
		return "audio/mpeg"
	case len(data) >= 2 && data[0] == 0xff && data[1]&0xe0 == 0xe0:
		return "audio/mpeg"
	case len(data) >= 4 && bytes.Equal(data[:4], []byte("OggS")):
		return "audio/ogg"
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WAVE")):
		return "audio/wav"
	case len(data) >= 4 && bytes.Equal(data[:4], []byte{0x1a, 0x45, 0xdf, 0xa3}):
		return "audio/webm"
	default:
		return ""
	}
}

func sameTranscriptionMediaType(actual, declared string) bool {
	if actual == declared {
		return true
	}
	switch actual {
	case "audio/mp4":
		return declared == "audio/m4a"
	case "audio/wav":
		return declared == "audio/x-wav"
	default:
		return false
	}
}

func readBoundedBody(reader io.Reader, limit int64) ([]byte, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("invalid response-body limit %d", limit)
	}
	data, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("response body exceeds %d bytes", limit)
	}
	return data, nil
}

func decodeBoundedJSON(reader io.Reader, limit int64, target any) error {
	data, err := readBoundedBody(reader, limit)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, target); err != nil {
		return err
	}
	return nil
}

// BuildEmbeddingsBatchLineBody is the OpenAI Batch API embeddings request
// codec. It lives with the OpenAI-compatible wire adapter rather than in rho's
// provider-neutral root package.
func BuildEmbeddingsBatchLineBody(req llm.EmbeddingRequest) (json.RawMessage, error) {
	if len(req.Input) == 0 {
		return nil, fmt.Errorf("openaicompat: embeddings require at least one input")
	}
	return json.Marshal(embeddingsRequestBody(req))
}

// ParseEmbeddingsBatchResultBody decodes an OpenAI-compatible embeddings result.
func ParseEmbeddingsBatchResultBody(
	raw json.RawMessage, expectedInputs int,
) (*llm.EmbeddingResponse, error) {
	var wire embeddingsAPIResponse
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("openaicompat: decode embeddings batch result: %w", err)
	}
	return wire.toResponse(expectedInputs)
}
