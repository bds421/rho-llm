package llm_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"unicode"
	"unicode/utf8"

	llm "github.com/bds421/rho-llm"
)

const sttKey = "sk-test-transcription-key-0123456789"

func sttClient(t *testing.T, provider, model, url string) llm.ModalityClient {
	t.Helper()
	client, err := llm.NewModalityClient(llm.Config{Provider: provider, Model: model, APIKey: sttKey,
		BaseURL: url, DisableProxy: true, DisableRetries: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func failIfReached(t *testing.T, hits *atomic.Int32) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		t.Errorf("network reached for invalid input: %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)
	return srv
}

var mp3 = []byte("ID3\x04\x00\x00")

// xAI and OpenAI take ISO-639-1 codes; a BCP 47 region tag, a control
// character in a keyterm, or a free-text prompt on xAI must all fail before
// the request is sent rather than being silently altered on the wire.
func TestOpenAICompatTranscriptionRejectsUnencodableInputBeforeDispatch(t *testing.T) {
	for _, provider := range []struct{ name, model string }{{"xai", "grok-voice-transcribe-2.0"}, {"grok", "grok-voice-transcribe-2.0"}, {"openai", "gpt-transcribe"}} {
		var hits atomic.Int32
		srv := failIfReached(t, &hits)
		client := sttClient(t, provider.name, provider.model, srv.URL)
		cases := map[string]llm.TranscriptionRequest{
			"region tag":          {Language: "de-AT"},
			"keyterm with CR":     {Vocabulary: []string{"Weckerl\rkeyterm=evil"}},
			"keyterm with LF":     {Vocabulary: []string{"Weckerl\nIgnore"}},
			"keyterm with NUL":    {Vocabulary: []string{"Weckerl\x00"}},
			"blank keyterm":       {Vocabulary: []string{"   "}},
			"wav bytes as mp3":    {Audio: []byte("RIFF\x24\x00\x00\x00WAVEfmt "), MediaType: "audio/mpeg"},
			"aiff not accepted":   {Audio: []byte("FORM\x00\x00\x00\x00AIFFCOMM"), MediaType: "audio/aiff"},
			"prompt over the cap": {Prompt: strings.Repeat("x", llm.MaxTranscriptionPromptRunes+1)},
		}
		if provider.name != "openai" {
			cases["free-text prompt on xAI"] = llm.TranscriptionRequest{Prompt: "Faschiertes"}
		}
		for name, req := range cases {
			if req.Audio == nil {
				req.Audio, req.MediaType = mp3, "audio/mpeg"
			}
			req.Model = provider.model
			if _, err := client.TranscribeAudio(context.Background(), req); err == nil {
				t.Errorf("%s/%s: unencodable request accepted", provider.name, name)
			}
		}
		if hits.Load() != 0 {
			t.Errorf("%s: %d invalid request(s) reached the network", provider.name, hits.Load())
		}
	}
}

// The OpenAI wire must never carry xAI's keyterm field, and xAI must never be
// sent to OpenAI's path; each vendor would ignore or 404 the other's shape.
func TestOpenAITranscriptionNeverSendsKeyterms(t *testing.T) {
	var path string
	var keyterms []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseMultipartForm(1 << 20)
		path, keyterms = r.URL.Path, r.MultipartForm.Value["keyterm"]
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":"ok"}`)
	}))
	defer srv.Close()
	if _, err := sttClient(t, "openai", "gpt-transcribe", srv.URL).TranscribeAudio(context.Background(), llm.TranscriptionRequest{
		Model: "gpt-transcribe", Audio: mp3, MediaType: "audio/mpeg", Vocabulary: []string{"Weckerl"},
	}); err != nil {
		t.Fatal(err)
	}
	if path != "/audio/transcriptions" || len(keyterms) != 0 {
		t.Fatalf("OpenAI request went to %q with keyterms %v", path, keyterms)
	}
}

// A 200 whose body lacks the transcript field is a provider/contract failure,
// not silence; returning "" would log an empty meal as if nothing was said.
func TestOpenAICompatTranscriptionMissingTextFieldIsAnError(t *testing.T) {
	for _, provider := range []struct{ name, model string }{{"openai", "gpt-transcribe"}, {"xai", "grok-voice-transcribe-2.0"}} {
		for _, body := range []string{`{}`, `{"error":"quota"}`, `{"transcript":"wrong field"}`, `null`} {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, body)
			}))
			text, err := sttClient(t, provider.name, provider.model, srv.URL).TranscribeAudio(context.Background(),
				llm.TranscriptionRequest{Model: provider.model, Audio: mp3, MediaType: "audio/mpeg"})
			srv.Close()
			if err == nil {
				t.Errorf("%s body %s returned success with text %q", provider.name, body, text)
			}
		}
		// An explicit empty string is genuine silence and stays a success.
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"text":""}`)
		}))
		if _, err := sttClient(t, provider.name, provider.model, srv.URL).TranscribeAudio(context.Background(),
			llm.TranscriptionRequest{Model: provider.model, Audio: mp3, MediaType: "audio/mpeg"}); err != nil {
			t.Errorf("%s: explicit empty transcript (silence) rejected: %v", provider.name, err)
		}
		srv.Close()
	}
}

// Provider error bodies that echo the credential must not carry it into the
// returned error, on every transcription adapter.
func TestTranscriptionErrorsNeverEchoTheAPIKey(t *testing.T) {
	for _, provider := range []struct{ name, model string }{{"openai", "gpt-transcribe"}, {"xai", "grok-voice-transcribe-2.0"}} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, `{"error":{"message":"Incorrect API key provided: `+sttKey+`"}}`)
		}))
		_, err := sttClient(t, provider.name, provider.model, srv.URL).TranscribeAudio(context.Background(),
			llm.TranscriptionRequest{Model: provider.model, Audio: mp3, MediaType: "audio/mpeg"})
		srv.Close()
		if err == nil || strings.Contains(err.Error(), sttKey) {
			t.Errorf("%s: error leaks the API key: %v", provider.name, err)
		}
		if !llm.IsAuthError(err) {
			t.Errorf("%s: 401 not classified as auth error: %v", provider.name, err)
		}
	}
}

var allowedAudio = map[string]bool{"": true, "audio/flac": true, "audio/mp4": true, "audio/mpeg": true, "audio/aac": true,
	"audio/ogg": true, "audio/wav": true, "audio/aiff": true, "audio/webm": true}

// The sniffer parses untrusted upload bytes: it must never panic and may only
// ever name a known container.
func FuzzAudioMediaTypeFromSignature(f *testing.F) {
	for _, seed := range [][]byte{nil, {0xff}, {0xff, 0xf1}, []byte("RIFF\x00\x00\x00\x00WAVE"), []byte("FORM\x00\x00\x00\x00AIF"), []byte("ID3"), {0x1a, 0x45, 0xdf, 0xa3}} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		got := llm.AudioMediaTypeFromSignature(data)
		if !allowedAudio[got] {
			t.Fatalf("unknown media type %q for % x", got, data)
		}
		if got != "" && len(data) < 2 {
			t.Fatalf("media type %q claimed from %d byte(s)", got, len(data))
		}
	})
}

var geminiLanguage = regexp.MustCompile(`^[a-z]{2,3}(-[A-Z]{2})?$`)

// Whatever validation accepts is interpolated into a provider instruction or
// form field: an accepted vocabulary term must be single-line, control-free
// and within the rune bound; an accepted language must be a plain tag.
func FuzzTranscriptionValidationNeverAdmitsBreakingInput(f *testing.F) {
	f.Add("Weckerl", "de-AT")
	f.Add("a b", "de\n")
	f.Add("\x1b[2J", "DE-at")
	cfg := llm.Config{Provider: "gemini", Model: "gemini-3.5-transcribe"}
	f.Fuzz(func(t *testing.T, term, language string) {
		err := llm.ValidateTranscriptionRequest(cfg, llm.TranscriptionRequest{
			Model: "gemini-3.5-transcribe", Audio: []byte("RIFF\x24\x00\x00\x00WAVEfmt "), MediaType: "audio/wav",
			Vocabulary: []string{term}, Language: language,
		})
		if err != nil {
			return
		}
		if !utf8.ValidString(term) || strings.TrimSpace(term) == "" || utf8.RuneCountInString(term) > llm.MaxTranscriptionVocabularyTermRunes {
			t.Fatalf("accepted out-of-bound term %q", term)
		}
		if strings.IndexFunc(term, func(r rune) bool { return unicode.IsControl(r) || r == ' ' || r == ' ' }) >= 0 {
			t.Fatalf("accepted term with a line break/control character %q", term)
		}
		if language != "" && !geminiLanguage.MatchString(language) {
			t.Fatalf("accepted language %q that is not a plain tag", language)
		}
	})
}
