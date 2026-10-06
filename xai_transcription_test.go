package llm_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// xAI speech-to-text is POST /stt with repeatable keyterm fields; sending it to
// OpenAI's /audio/transcriptions path would 404.
func TestXAITranscriptionUsesSTTEndpointAndKeyterms(t *testing.T) {
	var path, model, language, auth string
	var keyterms, prompts []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path, auth = r.URL.Path, r.Header.Get("Authorization")
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("multipart: %v", err)
		}
		model, language = r.FormValue("model"), r.FormValue("language")
		keyterms, prompts = r.MultipartForm.Value["keyterm"], r.MultipartForm.Value["prompt"]
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":"zwei Weckerl","language":"de","duration":1.2}`)
	}))
	defer srv.Close()

	cfg := llm.Config{Provider: "xai", Model: "grok-voice-transcribe-2.0", APIKey: "xai-key", BaseURL: srv.URL, DisableProxy: true}
	client, err := llm.NewModalityClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	text, err := client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{
		Model: "grok-voice-transcribe-2.0", Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/mpeg",
		Language: "de", Vocabulary: []string{"Weckerl", " ", "Meal Prep"},
	})
	if err == nil {
		t.Fatal("blank vocabulary term accepted")
	}
	text, err = client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{
		Model: "grok-voice-transcribe-2.0", Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/mpeg",
		Language: "de", Vocabulary: []string{"Weckerl", "Meal Prep"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != "zwei Weckerl" || path != "/stt" || model != "grok-voice-transcribe-2.0" || language != "de" {
		t.Fatalf("text=%q path=%q model=%q language=%q", text, path, model, language)
	}
	if auth != "Bearer xai-key" {
		t.Fatalf("auth header = %q", auth)
	}
	if strings.Join(keyterms, "|") != "Weckerl|Meal Prep" || len(prompts) != 0 {
		t.Fatalf("keyterms=%v prompts=%v; vocabulary must go to keyterm, never prompt", keyterms, prompts)
	}
	if _, err := client.TranscribeAudio(context.Background(), llm.TranscriptionRequest{
		Model: "grok-voice-transcribe-2.0", Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/mpeg", Prompt: "hint",
	}); err == nil {
		t.Fatal("xAI accepted a free-text prompt it cannot encode")
	}
}

// OpenAI folds Vocabulary into Whisper's prompt after any caller prompt.
func TestOpenAITranscriptionFoldsVocabularyIntoPrompt(t *testing.T) {
	var prompt string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseMultipartForm(1 << 20)
		prompt = r.FormValue("prompt")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":"ok"}`)
	}))
	defer srv.Close()
	if _, err := transcribeAudio(context.Background(), cfgFor(srv.URL), llm.TranscriptionRequest{
		Model: "gpt-transcribe", Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/mpeg",
		Prompt: "Food log.", Vocabulary: []string{"Weckerl", "Meal Prep"},
	}); err != nil {
		t.Fatal(err)
	}
	if prompt != "Food log. Weckerl, Meal Prep" {
		t.Fatalf("prompt = %q", prompt)
	}
}
