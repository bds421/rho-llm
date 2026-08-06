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
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

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
	maxBatchRequests   = 50_000
	maxBatchInputBytes = 200_000_000

	// OpenAI counts embedding inputs rather than JSONL request lines for this
	// separate limit. One embedding request can contain many inputs.
	maxBatchEmbeddingInputs = 50_000
)

const openAIMaxTurnaround = 24 * time.Hour

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
	clientConfig := cfg
	clientConfig.Timeout = timeout
	httpClient, err := llm.NewSafeHTTPClient(clientConfig)
	if err != nil {
		return nil, err
	}
	return &Client{
		cfg:          cfg,
		httpClient:   httpClient,
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
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	if opts.MaxTurnaround != openAIMaxTurnaround {
		return nil, fmt.Errorf("openaibatch: max turnaround must be exactly 24h")
	}
	if _, reserved := opts.Metadata[recoveryMetadataKey]; reserved {
		return nil, fmt.Errorf("openaibatch: metadata key %q is reserved", recoveryMetadataKey)
	}
	if _, reserved := opts.Metadata[requestDigestMetaKey]; reserved {
		return nil, fmt.Errorf("openaibatch: metadata key %q is reserved", requestDigestMetaKey)
	}
	if opts.RecoveryKey != "" && len(opts.Metadata)+2 > llm.MaxBatchMetadataEntries {
		return nil, fmt.Errorf("openaibatch: batch metadata leaves no room for recovery identity")
	}
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

	jsonl, err := buildBatchJSONL(k, items, endpoint, chatT, respT, maxBatchInputBytes)
	if err != nil {
		return nil, err
	}
	const window = "24h"
	digest := batchRequestDigest(endpoint, window, jsonl)
	if opts.RecoveryKey != "" {
		recovered, recoverErr := c.Recover(ctx, opts.RecoveryKey)
		switch {
		case recoverErr == nil:
			return requireRecoveredBatch(recovered, opts.RecoveryKey, digest, endpoint)
		case !errors.Is(recoverErr, llm.ErrBatchNotFound):
			return nil, recoverErr
		}
	}

	fileID, err := c.uploadInputFile(ctx, jsonl)
	if err != nil {
		return nil, err
	}

	metadata := make(map[string]string, len(opts.Metadata)+2)
	for key, value := range opts.Metadata {
		metadata[key] = value
	}
	if opts.RecoveryKey != "" {
		metadata[recoveryMetadataKey] = opts.RecoveryKey
		metadata[requestDigestMetaKey] = digest
	}
	handle, err := c.createBatch(ctx, fileID, endpoint, window, metadata)
	if err == nil {
		if opts.RecoveryKey == "" {
			return handle, nil
		}
		return requireRecoveredBatch(handle, opts.RecoveryKey, digest, endpoint)
	}
	if opts.RecoveryKey == "" {
		return nil, err
	}
	recovered, recoverErr := c.Recover(ctx, opts.RecoveryKey)
	if recoverErr == nil {
		return requireRecoveredBatch(recovered, opts.RecoveryKey, digest, endpoint)
	}
	if errors.Is(recoverErr, llm.ErrBatchNotFound) {
		return nil, err
	}
	return nil, errors.Join(err, recoverErr)
}

func batchRequestDigest(endpoint, window string, jsonl []byte) string {
	hash := sha256.New()
	_, _ = hash.Write([]byte("rho-llm-batch-request/v1\x00"))
	_, _ = hash.Write([]byte(endpoint))
	_, _ = hash.Write([]byte("\x00"))
	_, _ = hash.Write([]byte(window))
	_, _ = hash.Write([]byte("\x00"))
	_, _ = hash.Write(jsonl)
	return fmt.Sprintf("sha256:%x", hash.Sum(nil))
}

func requireRecoveredBatch(
	handle *llm.BatchHandle,
	recoveryKey, requestDigest, endpoint string,
) (*llm.BatchHandle, error) {
	if handle == nil || handle.Validate() != nil || handle.RecoveryKey != recoveryKey ||
		handle.RequestDigest != requestDigest {
		return nil, fmt.Errorf("%w: recovery key %q names a different request", llm.ErrBatchRecoveryConflict, recoveryKey)
	}
	state, err := decodeAdapterState(*handle, handle.Provider)
	if err != nil || state.Endpoint != endpoint {
		return nil, fmt.Errorf("%w: recovery key %q names a different request", llm.ErrBatchRecoveryConflict, recoveryKey)
	}
	return handle, nil
}

// batchLine is one JSONL entry in the uploaded input file.
type batchLine struct {
	CustomID string          `json:"custom_id"`
	Method   string          `json:"method"`
	URL      string          `json:"url"`
	Body     json.RawMessage `json:"body"`
}

// buildBatchJSONL owns the encoded-input limit and checks it after every line.
// maxBytes is injected only so small white-box fixtures can prove the exact
// cap/cap+1 behavior without allocating the provider's 200 MB allowance.
func buildBatchJSONL(
	k kind,
	items []llm.BatchItem,
	endpoint string,
	chatT *openaicompat.Client,
	respT *openairesponses.Client,
	maxBytes int,
) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf) // Encode appends '\n' → JSONL
	for _, it := range items {
		body, err := buildLineBody(k, it, chatT, respT)
		if err != nil {
			return nil, fmt.Errorf("openaibatch: build line %q: %w", it.ItemID, err)
		}
		if err := enc.Encode(batchLine{CustomID: it.ItemID, Method: "POST", URL: endpoint, Body: body}); err != nil {
			return nil, fmt.Errorf("openaibatch: encode line %q: %w", it.ItemID, err)
		}
		if err := validateBatchInputBytesLimit(buf.Len(), maxBytes); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

// buildLineBody renders one item's request body using the kind's codec.
func buildLineBody(k kind, it llm.BatchItem, chatT *openaicompat.Client, respT *openairesponses.Client) (json.RawMessage, error) {
	switch k {
	case kindChat:
		return chatT.BuildChatBatchLineBody(*it.Request)
	case kindResponses:
		return respT.BuildResponsesBatchLineBody(*it.Request)
	case kindEmbeddings:
		return openaicompat.BuildEmbeddingsBatchLineBody(*it.Embedding)
	default:
		return nil, fmt.Errorf("openaibatch: unknown batch kind")
	}
}

// inferKind validates every item and resolves the single homogeneous endpoint for the
// batch. It rejects an empty set, per-item invariant violations, duplicate custom_ids,
// and any mix of endpoints — all before the caller's first network effect.
func inferKind(cfg llm.Config, items []llm.BatchItem) (kind, error) {
	if err := validateBatchRequestCount(len(items)); err != nil {
		return kindUnknown, err
	}
	seen := make(map[string]struct{}, len(items))
	k := kindUnknown
	embeddingInputs := 0
	for _, it := range items {
		if err := it.Validate(); err != nil {
			return kindUnknown, err
		}
		if _, dup := seen[it.ItemID]; dup {
			return kindUnknown, fmt.Errorf("openaibatch: duplicate item_id %q (item_ids must be unique to correlate results)", it.ItemID)
		}
		seen[it.ItemID] = struct{}{}
		if it.Request != nil {
			if err := llm.ValidateRequestCapabilities(cfg, *it.Request, false); err != nil {
				return kindUnknown, fmt.Errorf("openaibatch: item %q: %w", it.ItemID, err)
			}
		} else {
			nextEmbeddingInputs, err := addBatchEmbeddingInputs(embeddingInputs, len(it.Embedding.Input))
			if err != nil {
				return kindUnknown, err
			}
			embeddingInputs = nextEmbeddingInputs
			if err := llm.RequireCapabilitiesForModel(cfg, it.Embedding.Model, llm.CapabilityEmbeddings); err != nil {
				return kindUnknown, fmt.Errorf("openaibatch: item %q: %w", it.ItemID, err)
			}
		}
		itemModel := ""
		if it.Request != nil {
			itemModel = it.Request.Model
		} else {
			itemModel = it.Embedding.Model
		}
		if err := llm.RequireCapabilitiesForModel(cfg, itemModel, llm.CapabilityBatch); err != nil {
			return kindUnknown, fmt.Errorf("openaibatch: item %q: %w", it.ItemID, err)
		}

		itemKind := kindEmbeddings
		if it.Request != nil {
			itemKind = chatOrResponses(cfg, it.Request)
		}
		if k == kindUnknown {
			k = itemKind
		} else if itemKind != k {
			return kindUnknown, fmt.Errorf(
				"openaibatch: batch is not homogeneous: item %q targets %s but the batch targets %s; OpenAI batches must use a single endpoint — split into separate batches",
				it.ItemID, endpointFor(itemKind), endpointFor(k))
		}
	}
	return k, nil
}

func validateBatchRequestCount(count int) error {
	if count <= 0 {
		return fmt.Errorf("openaibatch: no items to submit")
	}
	if count > maxBatchRequests {
		return fmt.Errorf("openaibatch: batch exceeds %d requests", maxBatchRequests)
	}
	return nil
}

func validateBatchInputBytes(size int) error {
	return validateBatchInputBytesLimit(size, maxBatchInputBytes)
}

func validateBatchInputBytesLimit(size, maxBytes int) error {
	if maxBytes <= 0 {
		return fmt.Errorf("openaibatch: invalid encoded JSONL limit")
	}
	if size <= 0 || size > maxBytes {
		return fmt.Errorf("openaibatch: encoded JSONL exceeds %d bytes", maxBytes)
	}
	return nil
}

// addBatchEmbeddingInputs is overflow-safe and enforces OpenAI's separate
// total-input limit across all embedding request lines.
func addBatchEmbeddingInputs(total, count int) (int, error) {
	if total < 0 || count < 0 || total > maxBatchEmbeddingInputs || count > maxBatchEmbeddingInputs-total {
		return 0, fmt.Errorf("openaibatch: batch exceeds %d embedding inputs", maxBatchEmbeddingInputs)
	}
	return total + count, nil
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
