package llm_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Whisper's `prompt` field must carry TranscriptionRequest.Prompt, and must be
// absent (not an empty field) when no prompt is given.
func TestOpenAITranscriptionForwardsPromptOnlyWhenSet(t *testing.T) {
	var prompts []*string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("parse multipart: %v", err)
		}
		if values, ok := r.MultipartForm.Value["prompt"]; ok {
			prompts = append(prompts, &values[0])
		} else {
			prompts = append(prompts, nil)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":"ok"}`)
	}))
	defer srv.Close()

	audio := []byte("ID3\x04\x00\x00")
	for _, prompt := range []string{"Faschiertes, Weckerl", ""} {
		if _, err := transcribeAudio(context.Background(), cfgFor(srv.URL), llm.TranscriptionRequest{
			Audio: audio, MediaType: "audio/mpeg", Prompt: prompt,
		}); err != nil {
			t.Fatalf("transcribe (prompt %q): %v", prompt, err)
		}
	}
	if len(prompts) != 2 || prompts[0] == nil || *prompts[0] != "Faschiertes, Weckerl" {
		t.Fatalf("prompt not forwarded: %v", prompts)
	}
	if prompts[1] != nil {
		t.Fatalf("empty prompt sent as field %q", *prompts[1])
	}
}

// The prompt bound is enforced provider-neutrally, before any adapter runs.
func TestTranscriptionPromptBoundIsProviderNeutral(t *testing.T) {
	long := make([]rune, llm.MaxTranscriptionPromptRunes+1)
	for i := range long {
		long[i] = 'x'
	}
	err := llm.ValidateTranscriptionRequest(llm.Config{Provider: "openai", Model: "whisper-1"}, llm.TranscriptionRequest{
		Model: "whisper-1", Audio: []byte("ID3\x04\x00\x00"), MediaType: "audio/mpeg", Prompt: string(long),
	})
	if err == nil {
		t.Fatal("over-long prompt accepted")
	}
}
