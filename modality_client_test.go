package llm_test

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

func TestModalityClientReusesPersistentSafeTransport(t *testing.T) {
	var connections atomic.Int32
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"text-embedding-3-small","data":[{"index":0,"embedding":[1]}]}`)
	}))
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			connections.Add(1)
		}
	}
	server.Start()
	defer server.Close()

	cfg := cfgFor(server.URL)
	cfg.Model = "text-embedding-3-small"
	client, err := llm.NewModalityClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	for range 2 {
		if _, err := client.GenerateEmbeddings(context.Background(), llm.EmbeddingRequest{
			Model: cfg.Model, Input: []string{"reuse"},
		}); err != nil {
			t.Fatal(err)
		}
	}
	if got := connections.Load(); got != 1 {
		t.Fatalf("connections=%d, want one reused connection", got)
	}
}

func TestModalityClientRejectsUnencodableRequestBeforeNetwork(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		requests.Add(1)
	}))
	defer server.Close()
	cfg := cfgFor(server.URL)
	cfg.Model = "gpt-image-1"
	client, err := llm.NewModalityClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	_, err = client.GenerateImages(context.Background(), llm.ImageRequest{
		Model: cfg.Model, Prompt: "fail closed", MediaType: "image/avif",
	})
	if err == nil {
		t.Fatal("unencodable output type was accepted")
	}
	if got := requests.Load(); got != 0 {
		t.Fatalf("provider requests=%d, want zero", got)
	}
}

func TestModalityClientHonorsInFlightCancellation(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		close(started)
		<-release
	}))
	defer server.Close()
	defer close(release)
	cfg := cfgFor(server.URL)
	cfg.Model = "text-embedding-3-small"
	client, err := llm.NewModalityClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, callErr := client.GenerateEmbeddings(ctx, llm.EmbeddingRequest{
			Model: cfg.Model, Input: []string{"cancel"},
		})
		done <- callErr
	}()
	select {
	case <-started:
		cancel()
	case <-time.After(2 * time.Second):
		t.Fatal("provider request did not start")
	}
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error=%v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled modality request did not return")
	}
}

func TestModalityClientRejectsCorruptEmbeddingResult(t *testing.T) {
	for name, response := range map[string]string{
		"missing vector":  `{"data":[]}`,
		"duplicate index": `{"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]}`,
		"empty vector":    `{"data":[{"index":0,"embedding":[]}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(w, response)
			}))
			defer server.Close()
			cfg := cfgFor(server.URL)
			cfg.Model = "text-embedding-3-small"
			client, err := llm.NewModalityClient(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer client.Close()
			inputs := []string{"x"}
			if name == "duplicate index" {
				inputs = append(inputs, "y")
			}
			if _, err := client.GenerateEmbeddings(context.Background(), llm.EmbeddingRequest{
				Model: cfg.Model, Input: inputs,
			}); err == nil {
				t.Fatal("corrupt embeddings response was accepted")
			}
		})
	}
}

func TestNewModalityClientFailsWithoutRegisteredProtocolDriver(t *testing.T) {
	_, err := llm.NewModalityClient(llm.Config{
		Provider: "anthropic", Model: "claude-sonnet-4-5", APIKey: "secret",
	})
	if err == nil {
		t.Fatal("anthropic modality client unexpectedly succeeded without a registered driver")
	}
}
