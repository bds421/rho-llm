package llm_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

func cfgFor(url string) llm.Config {
	return llm.Config{Provider: "openai", BaseURL: url, APIKey: "k", MaxTokens: 10}
}

func generateEmbeddings(
	ctx context.Context, cfg llm.Config, req llm.EmbeddingRequest,
) (*llm.EmbeddingResponse, error) {
	if req.Model == "" {
		req.Model = "text-embedding-3-small"
	}
	cfg.Model = req.Model
	client, err := llm.NewModalityClient(cfg)
	if err != nil {
		return nil, err
	}
	defer client.Close()
	return client.GenerateEmbeddings(ctx, req)
}

func generateImages(
	ctx context.Context, cfg llm.Config, req llm.ImageRequest,
) (*llm.ImageResponse, error) {
	if req.Model == "" {
		req.Model = "gpt-image-1"
	}
	cfg.Model = req.Model
	client, err := llm.NewModalityClient(cfg)
	if err != nil {
		return nil, err
	}
	defer client.Close()
	return client.GenerateImages(ctx, req)
}

func synthesizeSpeech(
	ctx context.Context, cfg llm.Config, req llm.SpeechRequest,
) (*llm.SpeechResponse, error) {
	if req.Model == "" {
		req.Model = "tts-1"
	}
	cfg.Model = req.Model
	client, err := llm.NewModalityClient(cfg)
	if err != nil {
		return nil, err
	}
	defer client.Close()
	return client.SynthesizeSpeech(ctx, req)
}

func transcribeAudio(
	ctx context.Context, cfg llm.Config, req llm.TranscriptionRequest,
) (string, error) {
	if req.Model == "" {
		req.Model = "whisper-1"
	}
	cfg.Model = req.Model
	client, err := llm.NewModalityClient(cfg)
	if err != nil {
		return "", err
	}
	defer client.Close()
	return client.TranscribeAudio(ctx, req)
}

func TestGenerateEmbeddings(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %q", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"model":"emb","data":[{"index":0,"embedding":[0.1,0.2]},{"index":1,"embedding":[0.3]}],"usage":{"prompt_tokens":7}}`)
	}))
	defer srv.Close()

	res, err := generateEmbeddings(context.Background(), cfgFor(srv.URL), llm.EmbeddingRequest{Model: "text-embedding-3-small", Input: []string{"a", "b"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Embeddings) != 2 || res.Embeddings[0].Vector[1] != 0.2 || res.InputTokens != 7 {
		t.Errorf("parsed wrong: %+v", res)
	}
	if gotBody["model"] != "text-embedding-3-small" {
		t.Errorf("request body model = %v", gotBody["model"])
	}

	// empty input → error, no request
	if _, err := generateEmbeddings(context.Background(), cfgFor(srv.URL), llm.EmbeddingRequest{Model: "text-embedding-3-small"}); err == nil {
		t.Error("empty input should error")
	}
}

func TestGenerateEmbeddingsUsesConfiguredProxy(t *testing.T) {
	var proxyCalls atomic.Int32
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxyCalls.Add(1)
		if r.URL.Host != "provider.invalid" || r.URL.Path != "/v1/embeddings" {
			t.Errorf("proxied request URL = %q", r.URL.String())
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"model":"text-embedding-3-small","data":[{"index":0,"embedding":[1]}]}`)
	}))
	defer proxy.Close()

	cfg := cfgFor("http://provider.invalid/v1")
	cfg.ProxyURL = proxy.URL
	result, err := generateEmbeddings(context.Background(), cfg, llm.EmbeddingRequest{
		Model: "text-embedding-3-small", Input: []string{"x"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 || proxyCalls.Load() != 1 {
		t.Fatalf("embeddings=%d proxy calls=%d, want 1 and 1", len(result.Embeddings), proxyCalls.Load())
	}
}

func TestGenerateImages(t *testing.T) {
	var gotBody map[string]any
	png := base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"data":[{"b64_json":"`+png+`"}]}`)
	}))
	defer srv.Close()

	res, err := generateImages(context.Background(), cfgFor(srv.URL), llm.ImageRequest{Model: "gpt-image-1", Prompt: "a cat", WidthPixels: 1024, HeightPixels: 1024, MediaType: "image/png"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Images) != 1 || res.Images[0].B64JSON != png ||
		res.Images[0].MediaType != "image/png" {
		t.Errorf("parsed wrong: %+v", res)
	}
	if gotBody["response_format"] != "b64_json" || gotBody["size"] != "1024x1024" ||
		gotBody["output_format"] != "png" {
		t.Errorf("request body wrong: %+v", gotBody)
	}
	if _, err := generateImages(context.Background(), cfgFor(srv.URL), llm.ImageRequest{Prompt: ""}); err == nil {
		t.Error("empty prompt should error")
	}
	if _, err := generateImages(context.Background(), cfgFor(srv.URL), llm.ImageRequest{
		Model: "gpt-image-1", Prompt: "a cat", WidthPixels: 100, MediaType: "image/png",
	}); err == nil {
		t.Error("partial geometry should error")
	}
	if _, err := generateImages(context.Background(), cfgFor(srv.URL), llm.ImageRequest{
		Model: "gpt-image-1", Prompt: "a cat", WidthPixels: 100, HeightPixels: 100,
		MediaType: "image/tiff",
	}); err == nil {
		t.Error("adapter-unsupported media type should error")
	}
}

func TestGenerateImagesVerifiesPayloadSignaturesAndLeavesURLUnlabeled(t *testing.T) {
	formats := []struct {
		name      string
		mediaType string
		payload   []byte
	}{
		{"png", "image/png", []byte("\x89PNG\r\n\x1a\n")},
		{"jpeg", "image/jpeg", []byte{0xff, 0xd8, 0xff, 0xe0}},
		{"webp", "image/webp", []byte("RIFF\x04\x00\x00\x00WEBP")},
	}
	for _, test := range formats {
		t.Run(test.name, func(t *testing.T) {
			encoded := base64.StdEncoding.EncodeToString(test.payload)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				io.WriteString(w, `{"data":[{"b64_json":"`+encoded+`"}]}`)
			}))
			defer server.Close()
			response, err := generateImages(context.Background(), cfgFor(server.URL), llm.ImageRequest{
				Model: "gpt-image-1", Prompt: "verified", MediaType: test.mediaType,
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(response.Images) != 1 || response.Images[0].MediaType != test.mediaType {
				t.Fatalf("response=%+v", response)
			}
		})
	}

	t.Run("mismatch", func(t *testing.T) {
		jpeg := base64.StdEncoding.EncodeToString([]byte{0xff, 0xd8, 0xff, 0xe0})
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			io.WriteString(w, `{"data":[{"b64_json":"`+jpeg+`"}]}`)
		}))
		defer server.Close()
		if _, err := generateImages(context.Background(), cfgFor(server.URL), llm.ImageRequest{
			Model: "gpt-image-1", Prompt: "mismatch", MediaType: "image/png",
		}); err == nil {
			t.Fatal("JPEG payload was labeled as requested PNG")
		}
	})

	t.Run("url only", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			io.WriteString(w, `{"data":[{"url":"https://artifacts.example/image"}]}`)
		}))
		defer server.Close()
		_, err := generateImages(context.Background(), cfgFor(server.URL), llm.ImageRequest{
			Model: "gpt-image-1", Prompt: "url", MediaType: "image/png",
		})
		if err == nil {
			t.Fatal("URL-only image was accepted after b64_json was required")
		}
	})

	t.Run("wrong result count", func(t *testing.T) {
		png := base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n"))
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			io.WriteString(w, `{"data":[{"b64_json":"`+png+`"}]}`)
		}))
		defer server.Close()
		_, err := generateImages(context.Background(), cfgFor(server.URL), llm.ImageRequest{
			Model: "gpt-image-1", Prompt: "two", N: 2, MediaType: "image/png",
		})
		if err == nil {
			t.Fatal("provider result count different from requested count was accepted")
		}
	})
}

func TestSynthesizeSpeech(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "audio/mpeg")
		w.Write([]byte("ID3\x04\x00\x00"))
	}))
	defer srv.Close()

	audio, err := synthesizeSpeech(context.Background(), cfgFor(srv.URL), llm.SpeechRequest{Model: "tts-1", Input: "hello", Voice: "alloy", MediaType: "audio/mpeg"})
	if err != nil {
		t.Fatal(err)
	}
	if string(audio.Audio) != "ID3\x04\x00\x00" || audio.MediaType != "audio/mpeg" {
		t.Errorf("audio = %+v", audio)
	}
	if gotBody["response_format"] != "mp3" || gotBody["voice"] != "alloy" {
		t.Errorf("request body wrong: %+v", gotBody)
	}
	if _, err := synthesizeSpeech(context.Background(), cfgFor(srv.URL), llm.SpeechRequest{Input: ""}); err == nil {
		t.Error("empty input should error")
	}
}

func TestSynthesizeSpeechVerifiesHeaderAndBoundedPayloadSignature(t *testing.T) {
	formats := []struct {
		name, requested, header, want string
		payload                       []byte
	}{
		{"mp3", "audio/mpeg", "audio/mpeg", "audio/mpeg", []byte("ID3\x04\x00\x00")},
		{"ogg opus", "audio/ogg; codecs=opus", "audio/opus", "audio/ogg; codecs=opus", []byte("OggS\x00\x00\x00\x00OpusHead")},
		{"aac", "audio/aac", "audio/aac", "audio/aac", []byte{0xff, 0xf1, 0x50, 0x80}},
		{"flac generic header", "audio/flac", "application/octet-stream", "audio/flac", []byte("fLaC\x00")},
		{"wav", "audio/wav", "audio/x-wav", "audio/wav", []byte("RIFF\x04\x00\x00\x00WAVE")},
		{"raw l16 exact header", "audio/L16", "audio/L16", "audio/L16", []byte{0, 1, 2, 3}},
	}
	for _, test := range formats {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", test.header)
				w.Write(test.payload)
			}))
			defer server.Close()
			response, err := synthesizeSpeech(context.Background(), cfgFor(server.URL), llm.SpeechRequest{
				Model: "tts-1", Input: "verified", Voice: "alloy", MediaType: test.requested,
			})
			if err != nil {
				t.Fatal(err)
			}
			if response.MediaType != test.want || string(response.Audio) != string(test.payload) {
				t.Fatalf("response=%+v", response)
			}
		})
	}

	for _, test := range []struct {
		name, requested, header string
		payload                 []byte
	}{
		{"header payload mismatch", "audio/mpeg", "audio/mpeg", []byte("RIFF\x04\x00\x00\x00WAVE")},
		{"requested payload mismatch", "audio/mpeg", "audio/wav", []byte("RIFF\x04\x00\x00\x00WAVE")},
		{"unknown payload", "audio/mpeg", "audio/mpeg", []byte("not audio")},
		{"raw l16 generic header", "audio/L16", "application/octet-stream", []byte{0, 1, 2, 3}},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", test.header)
				w.Write(test.payload)
			}))
			defer server.Close()
			if _, err := synthesizeSpeech(context.Background(), cfgFor(server.URL), llm.SpeechRequest{
				Model: "tts-1", Input: "reject", Voice: "alloy", MediaType: test.requested,
			}); err == nil {
				t.Fatal("unverified or mismatched speech output was accepted")
			}
		})
	}
}

func TestNonChatRequestValidatorsPerformNoNetworkIO(t *testing.T) {
	cfg := llm.Config{Provider: "openai", BaseURL: "http://127.0.0.1:1", APIKey: "k"}
	if err := llm.ValidateEmbeddingRequest(cfg, llm.EmbeddingRequest{
		Model: "text-embedding-3-small", Input: []string{"probe"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := llm.ValidateImageRequest(cfg, llm.ImageRequest{
		Model: "gpt-image-1", Prompt: "probe", MediaType: "image/png",
	}); err != nil {
		t.Fatal(err)
	}
	if err := llm.ValidateSpeechRequest(cfg, llm.SpeechRequest{
		Model: "tts-1", Input: "probe", Voice: "alloy", MediaType: "audio/mpeg",
	}); err != nil {
		t.Fatal(err)
	}
	if err := llm.ValidateSpeechRequest(cfg, llm.SpeechRequest{
		Model: "tts-1", Input: "probe", MediaType: "audio/mpeg",
	}); err == nil {
		t.Fatal("speech request without an app-selected voice was accepted")
	}
	for _, alias := range []string{"audio/opus", "audio/ogg", "audio/x-wav", "audio/l16"} {
		if err := llm.ValidateSpeechRequest(cfg, llm.SpeechRequest{
			Model: "tts-1", Input: "probe", Voice: "alloy", MediaType: alias,
		}); err == nil {
			t.Fatalf("non-canonical speech request media type %q was accepted", alias)
		}
	}
	if err := llm.ValidateTranscriptionRequest(cfg, llm.TranscriptionRequest{
		Model: "whisper-1", Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/mpeg", Language: "de",
	}); err != nil {
		t.Fatal(err)
	}
	for _, language := range []string{"de-AT", "DE", "eng"} {
		if err := llm.ValidateTranscriptionRequest(cfg, llm.TranscriptionRequest{
			Model: "whisper-1", Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/mpeg", Language: language,
		}); err == nil {
			t.Fatalf("unencodable transcription language %q was accepted", language)
		}
	}
}

func TestTranscriptionValidationChecksEverySupportedAudioSignature(t *testing.T) {
	cfg := llm.Config{Provider: "openai", Model: "whisper-1", APIKey: "k"}
	for _, test := range []struct {
		mediaType string
		data      []byte
	}{
		{"audio/flac", []byte("fLaC\x00")},
		{"audio/m4a", []byte{0, 0, 0, 12, 'f', 't', 'y', 'p', 'M', '4', 'A', ' '}},
		{"audio/mp4", []byte{0, 0, 0, 12, 'f', 't', 'y', 'p', 'i', 's', 'o', 'm'}},
		{"audio/mpeg", []byte("ID3\x04\x00\x00")},
		{"audio/ogg", []byte("OggS\x00")},
		{"audio/wav", []byte("RIFF\x04\x00\x00\x00WAVE")},
		{"audio/x-wav", []byte("RIFF\x04\x00\x00\x00WAVE")},
		{"audio/webm", []byte{0x1a, 0x45, 0xdf, 0xa3}},
	} {
		err := llm.ValidateTranscriptionRequest(cfg, llm.TranscriptionRequest{
			Model: "whisper-1", Audio: test.data, MediaType: test.mediaType,
		})
		if err != nil {
			t.Errorf("%s: %v", test.mediaType, err)
		}
	}
	for _, test := range []llm.TranscriptionRequest{
		{Model: "whisper-1", Audio: []byte("not audio"), MediaType: "audio/mpeg"},
		{Model: "whisper-1", Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/wav"},
	} {
		if err := llm.ValidateTranscriptionRequest(cfg, test); err == nil {
			t.Fatalf("mislabeled transcription input was accepted: %+v", test)
		}
	}
}

func TestTranscribeAudio(t *testing.T) {
	var gotModel, gotFile string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("parse multipart: %v", err)
		}
		gotModel = r.FormValue("model")
		f, hdr, err := r.FormFile("file")
		if err == nil {
			defer f.Close()
			gotFile = hdr.Filename
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"text":"transcribed text"}`)
	}))
	defer srv.Close()

	text, err := transcribeAudio(context.Background(), cfgFor(srv.URL), llm.TranscriptionRequest{
		Model: "whisper-1", Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/mpeg",
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != "transcribed text" {
		t.Errorf("text = %q", text)
	}
	if gotModel != "whisper-1" || gotFile != "audio.mp3" {
		t.Errorf("multipart fields wrong: model=%q file=%q", gotModel, gotFile)
	}
	if _, err := transcribeAudio(context.Background(), cfgFor(srv.URL), llm.TranscriptionRequest{Model: "whisper-1"}); err == nil {
		t.Error("empty audio should error")
	}
	if _, err := transcribeAudio(context.Background(), cfgFor(srv.URL), llm.TranscriptionRequest{
		Model: "whisper-1", Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/unknown",
	}); err == nil {
		t.Error("adapter-unsupported media type should error")
	}
}

func TestCapabilitiesErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		io.WriteString(w, `{"error":{"message":"bad key"}}`)
	}))
	defer srv.Close()
	_, err := generateEmbeddings(context.Background(), cfgFor(srv.URL), llm.EmbeddingRequest{Model: "text-embedding-3-small", Input: []string{"x"}})
	if !llm.IsAuthError(err) {
		t.Errorf("expected auth error, got %v", err)
	}
}

func TestCapabilitiesRespectBlockPrivateBaseURL(t *testing.T) {
	cfg := llm.Config{Provider: "openai", BaseURL: "http://169.254.169.254/v1", APIKey: "k", BlockPrivateBaseURL: true}
	if _, err := generateEmbeddings(context.Background(), cfg, llm.EmbeddingRequest{Model: "text-embedding-3-small", Input: []string{"x"}}); err == nil {
		t.Error("BlockPrivateBaseURL should reject embeddings to a private host")
	}
}

func TestNonChatCapabilitiesUseRetryPolicyAndHook(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			io.WriteString(w, `{"error":{"message":"try again"}}`)
			return
		}
		io.WriteString(w, `{"model":"text-embedding-3-small","data":[{"index":0,"embedding":[1]}]}`)
	}))
	defer srv.Close()
	var hookCalls atomic.Int32
	cfg := cfgFor(srv.URL)
	cfg.MaxRetries = 3
	cfg.RetryPolicy = &llm.RetryPolicy{BaseDelay: time.Microsecond, MaxDelay: time.Microsecond, Factor: 1}
	cfg.RetryHook = func(llm.RetryEvent) { hookCalls.Add(1) }
	if _, err := generateEmbeddings(context.Background(), cfg, llm.EmbeddingRequest{
		Model: "text-embedding-3-small", Input: []string{"x"},
	}); err != nil {
		t.Fatalf("GenerateEmbeddings: %v", err)
	}
	if calls.Load() != 2 || hookCalls.Load() < 2 {
		t.Fatalf("calls=%d hookCalls=%d", calls.Load(), hookCalls.Load())
	}
}

func TestNonChatCapabilitiesCanDisableHiddenRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		io.WriteString(w, `{"error":{"message":"try again"}}`)
	}))
	defer srv.Close()
	cfg := cfgFor(srv.URL)
	cfg.DisableRetries = true
	_, err := generateEmbeddings(context.Background(), cfg, llm.EmbeddingRequest{
		Model: "text-embedding-3-small", Input: []string{"x"},
	})
	if err == nil {
		t.Fatal("GenerateEmbeddings unexpectedly succeeded")
	}
	if calls.Load() != 1 {
		t.Fatalf("provider calls=%d, want exactly one", calls.Load())
	}
}
