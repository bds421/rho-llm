package llm_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func cfgFor(url string) llm.Config {
	return llm.Config{Provider: "openai", BaseURL: url, APIKey: "k", MaxTokens: 10}
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

	res, err := llm.GenerateEmbeddings(context.Background(), cfgFor(srv.URL), llm.EmbeddingRequest{Model: "emb", Input: []string{"a", "b"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Embeddings) != 2 || res.Embeddings[0].Vector[1] != 0.2 || res.InputTokens != 7 {
		t.Errorf("parsed wrong: %+v", res)
	}
	if gotBody["model"] != "emb" {
		t.Errorf("request body model = %v", gotBody["model"])
	}

	// empty input → error, no request
	if _, err := llm.GenerateEmbeddings(context.Background(), cfgFor(srv.URL), llm.EmbeddingRequest{Model: "emb"}); err == nil {
		t.Error("empty input should error")
	}
}

func TestGenerateImages(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"data":[{"b64_json":"AAAA"}]}`)
	}))
	defer srv.Close()

	res, err := llm.GenerateImages(context.Background(), cfgFor(srv.URL), llm.ImageRequest{Model: "img", Prompt: "a cat", Size: "1024x1024"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Images) != 1 || res.Images[0].B64JSON != "AAAA" {
		t.Errorf("parsed wrong: %+v", res)
	}
	if gotBody["response_format"] != "b64_json" || gotBody["size"] != "1024x1024" {
		t.Errorf("request body wrong: %+v", gotBody)
	}
	if _, err := llm.GenerateImages(context.Background(), cfgFor(srv.URL), llm.ImageRequest{Prompt: ""}); err == nil {
		t.Error("empty prompt should error")
	}
}

func TestSynthesizeSpeech(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/mpeg")
		w.Write([]byte("RAWAUDIO"))
	}))
	defer srv.Close()

	audio, err := llm.SynthesizeSpeech(context.Background(), cfgFor(srv.URL), llm.SpeechRequest{Model: "tts", Input: "hello", Voice: "alloy"})
	if err != nil {
		t.Fatal(err)
	}
	if string(audio) != "RAWAUDIO" {
		t.Errorf("audio = %q", audio)
	}
	if _, err := llm.SynthesizeSpeech(context.Background(), cfgFor(srv.URL), llm.SpeechRequest{Input: ""}); err == nil {
		t.Error("empty input should error")
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

	text, err := llm.TranscribeAudio(context.Background(), cfgFor(srv.URL), llm.TranscriptionRequest{
		Model: "whisper", Audio: []byte("fakeaudio"), Filename: "clip.mp3",
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != "transcribed text" {
		t.Errorf("text = %q", text)
	}
	if gotModel != "whisper" || gotFile != "clip.mp3" {
		t.Errorf("multipart fields wrong: model=%q file=%q", gotModel, gotFile)
	}
	if _, err := llm.TranscribeAudio(context.Background(), cfgFor(srv.URL), llm.TranscriptionRequest{Model: "w"}); err == nil {
		t.Error("empty audio should error")
	}
}

func TestCapabilitiesErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		io.WriteString(w, `{"error":{"message":"bad key"}}`)
	}))
	defer srv.Close()
	_, err := llm.GenerateEmbeddings(context.Background(), cfgFor(srv.URL), llm.EmbeddingRequest{Model: "e", Input: []string{"x"}})
	if !llm.IsAuthError(err) {
		t.Errorf("expected auth error, got %v", err)
	}
}

func TestCapabilitiesRespectBlockPrivateBaseURL(t *testing.T) {
	cfg := llm.Config{Provider: "openai", BaseURL: "http://169.254.169.254/v1", APIKey: "k", BlockPrivateBaseURL: true}
	if _, err := llm.GenerateEmbeddings(context.Background(), cfg, llm.EmbeddingRequest{Model: "e", Input: []string{"x"}}); err == nil {
		t.Error("BlockPrivateBaseURL should reject embeddings to a private host")
	}
}
