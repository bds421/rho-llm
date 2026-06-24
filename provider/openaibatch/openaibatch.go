// Package openaibatch implements the OpenAI Batch API (asynchronous, bulk request
// processing at ~50% cost) as an llm.BatchClient. It is the batch counterpart to the
// synchronous openaicompat/openairesponses adapters and reuses their wire translation
// verbatim (see each package's batchline.go) so a batched line is byte-identical to
// what a live Complete call would send.
//
// Transport: OpenAI batches go through the Files API — upload a JSONL file, create a
// batch referencing it, poll, then download the result file (files.go + rest.go). A
// single batch targets exactly one endpoint, so Submit validates that all items are
// homogeneous (all chat, all responses, or all embeddings) before any upload.
package openaibatch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

func init() {
	factory := func(cfg llm.Config) (llm.BatchClient, error) { return New(cfg) }
	// The same driver handles both OpenAI wire protocols; the per-item endpoint is
	// inferred from the items, not from which protocol key resolved the driver.
	llm.RegisterBatchProvider("openai_compat", factory)
	llm.RegisterBatchProvider("openai_responses", factory)
}

// Absolute OpenAI API paths used both as the batch `endpoint` and each JSONL line's
// `url`. These are independent of the Files/Batches base URL (which already ends in
// /v1) — OpenAI requires the absolute "/v1/..." path here.
const (
	endpointChat       = "/v1/chat/completions"
	endpointResponses  = "/v1/responses"
	endpointEmbeddings = "/v1/embeddings"
)

// kind is the homogeneous request type of a batch (OpenAI allows one endpoint per batch).
type kind int

const (
	kindUnknown kind = iota
	kindChat
	kindResponses
	kindEmbeddings
)

func endpointFor(k kind) string {
	switch k {
	case kindChat:
		return endpointChat
	case kindResponses:
		return endpointResponses
	case kindEmbeddings:
		return endpointEmbeddings
	default:
		return ""
	}
}

// Client is the OpenAI Batch API driver. All fields are set once in New and read-only
// thereafter; each method builds its own translators locally, so concurrent use is
// race-free without a mutex.
type Client struct {
	cfg          llm.Config
	httpClient   *http.Client
	baseURL      string // resolved Files/Batches base, e.g. "https://api.openai.com/v1"
	authHeader   string // "Bearer" (or "" for no auth)
	providerName string
}

var _ llm.BatchClient = (*Client)(nil)

// New creates an OpenAI Batch API client. It performs no I/O.
func New(cfg llm.Config) (*Client, error) {
	if cfg.APIKey == "" && !llm.IsNoAuthProvider(cfg.Provider) {
		return nil, fmt.Errorf("openaibatch: %s API key is required", cfg.Provider)
	}
	baseURL := llm.ResolveBaseURL(cfg)
	if baseURL == "" {
		return nil, fmt.Errorf("openaibatch: no base URL configured for provider %s (set BaseURL in config)", cfg.Provider)
	}
	providerName := cfg.ProviderName
	if providerName == "" {
		providerName = cfg.Provider
	}
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = llm.DefaultTimeout
	}
	return &Client{
		cfg:          cfg,
		httpClient:   llm.SafeHTTPClient(timeout),
		baseURL:      strings.TrimRight(baseURL, "/"),
		authHeader:   llm.ResolveAuthHeader(cfg),
		providerName: providerName,
	}, nil
}

// Close releases idle HTTP connections.
func (c *Client) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}

// Submit validates the items, assembles a JSONL file, uploads it via the Files API,
// and creates the batch. It fails before any network effect on an invalid set
// (empty, per-item invariant violation, duplicate custom_id, or mixed endpoints).
func (c *Client) Submit(ctx context.Context, items []llm.BatchItem, opts llm.BatchOptions) (*llm.BatchHandle, error) {
	k, err := inferKind(c.cfg, items)
	if err != nil {
		return nil, err
	}
	endpoint := endpointFor(k)

	// Build the codec once per batch (network-free) and reuse it for every line.
	var chatT *openaicompat.Client
	var respT *openairesponses.Client
	switch k {
	case kindChat:
		if chatT, err = openaicompat.BatchTranslator(c.cfg); err != nil {
			return nil, err
		}
	case kindResponses:
		if respT, err = openairesponses.BatchTranslator(c.cfg); err != nil {
			return nil, err
		}
	}

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf) // Encode appends '\n' → JSONL
	for _, it := range items {
		body, err := buildLineBody(k, it, chatT, respT)
		if err != nil {
			return nil, fmt.Errorf("openaibatch: build line %q: %w", it.CustomID, err)
		}
		if err := enc.Encode(batchLine{CustomID: it.CustomID, Method: "POST", URL: endpoint, Body: body}); err != nil {
			return nil, fmt.Errorf("openaibatch: encode line %q: %w", it.CustomID, err)
		}
	}

	fileID, err := c.uploadInputFile(ctx, buf.Bytes())
	if err != nil {
		return nil, err
	}

	window := opts.CompletionWindow
	if window == "" {
		window = llm.DefaultCompletionWindow
	}
	return c.createBatch(ctx, fileID, endpoint, window, opts.Metadata)
}

// batchLine is one JSONL entry in the uploaded input file.
type batchLine struct {
	CustomID string          `json:"custom_id"`
	Method   string          `json:"method"`
	URL      string          `json:"url"`
	Body     json.RawMessage `json:"body"`
}

// buildLineBody renders one item's request body using the kind's codec.
func buildLineBody(k kind, it llm.BatchItem, chatT *openaicompat.Client, respT *openairesponses.Client) (json.RawMessage, error) {
	switch k {
	case kindChat:
		return chatT.BuildChatBatchLineBody(*it.Request)
	case kindResponses:
		return respT.BuildResponsesBatchLineBody(*it.Request)
	case kindEmbeddings:
		return llm.BuildEmbeddingsBatchLineBody(*it.Embedding)
	default:
		return nil, fmt.Errorf("openaibatch: unknown batch kind")
	}
}

// inferKind validates every item and resolves the single homogeneous endpoint for the
// batch. It rejects an empty set, per-item invariant violations, duplicate custom_ids,
// and any mix of endpoints — all before the caller's first network effect.
func inferKind(cfg llm.Config, items []llm.BatchItem) (kind, error) {
	if len(items) == 0 {
		return kindUnknown, fmt.Errorf("openaibatch: no items to submit")
	}
	seen := make(map[string]struct{}, len(items))
	k := kindUnknown
	for _, it := range items {
		if err := it.Validate(); err != nil {
			return kindUnknown, err
		}
		if _, dup := seen[it.CustomID]; dup {
			return kindUnknown, fmt.Errorf("openaibatch: duplicate custom_id %q (custom_ids must be unique to correlate results)", it.CustomID)
		}
		seen[it.CustomID] = struct{}{}

		itemKind := kindEmbeddings
		if it.Request != nil {
			itemKind = chatOrResponses(cfg, it.Request)
		}
		if k == kindUnknown {
			k = itemKind
		} else if itemKind != k {
			return kindUnknown, fmt.Errorf(
				"openaibatch: batch is not homogeneous: item %q targets %s but the batch targets %s; OpenAI batches must use a single endpoint — split into separate batches",
				it.CustomID, endpointFor(itemKind), endpointFor(k))
		}
	}
	return k, nil
}

// chatOrResponses routes a chat Request to the chat or responses endpoint using the
// exact same rule as a live call (llm.ResolveProtocol with the item's model), so a
// GPT-5 item batches to /v1/responses just as Complete would send it there.
func chatOrResponses(cfg llm.Config, req *llm.Request) kind {
	c := cfg
	if req.Model != "" {
		c.Model = req.Model
	}
	if llm.ResolveProtocol(c) == "openai_responses" {
		return kindResponses
	}
	return kindChat
}
