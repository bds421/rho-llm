// Package geminibatch implements the Gemini Batch API as an llm.BatchClient.
// Uses inline requests (under 20MB) for completion and embedding batches.
package geminibatch

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/gemini"
)

func init() {
	llm.RegisterBatchProvider("gemini", func(cfg llm.Config) (llm.BatchClient, error) {
		return New(cfg)
	})
}

const (
	geminiMaxTurnaround = 24 * time.Hour
	adapterStateVersion = 1
	maxInlineBytes      = 20 << 20
)

// Client is the Gemini Batch driver.
type Client struct {
	cfg          llm.Config
	httpClient   *http.Client
	baseURL      string // e.g. https://generativelanguage.googleapis.com/v1beta
	providerName string
	translator   *gemini.Client
}

var _ llm.BatchClient = (*Client)(nil)

// New creates a Gemini batch client.
func New(cfg llm.Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("geminibatch: API key is required")
	}
	modelsBase := llm.ResolveBaseURL(cfg)
	if modelsBase == "" {
		modelsBase = "https://generativelanguage.googleapis.com/v1beta/models"
	}
	// Strip trailing /models for resource paths like /v1beta/batches/...
	base := strings.TrimSuffix(strings.TrimRight(modelsBase, "/"), "/models")
	providerName := cfg.ProviderName
	if providerName == "" {
		providerName = cfg.Provider
		if providerName == "" {
			providerName = "gemini"
		}
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
	translator, err := gemini.BatchTranslator(cfg)
	if err != nil {
		return nil, err
	}
	return &Client{
		cfg:          cfg,
		httpClient:   httpClient,
		baseURL:      base,
		providerName: providerName,
		translator:   translator,
	}, nil
}

// Close releases idle connections.
func (c *Client) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}

// Submit creates an inline batch for either all completion or all embedding items.
func (c *Client) Submit(ctx context.Context, items []llm.BatchItem, opts llm.BatchOptions) (*llm.BatchHandle, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	if opts.MaxTurnaround != geminiMaxTurnaround {
		return nil, fmt.Errorf("geminibatch: max turnaround must be exactly 24h")
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("geminibatch: batch has no items")
	}
	kind, model, err := inferKind(items)
	if err != nil {
		return nil, err
	}
	var inline []map[string]any
	seen := make(map[string]struct{}, len(items))
	for _, item := range items {
		if err := item.Validate(); err != nil {
			return nil, err
		}
		if _, dup := seen[item.ItemID]; dup {
			return nil, fmt.Errorf("geminibatch: duplicate item_id %q", item.ItemID)
		}
		seen[item.ItemID] = struct{}{}
		switch kind {
		case llm.BatchOperationCompletion:
			body, err := c.translator.BuildGenerateContentBody(*item.Request)
			if err != nil {
				return nil, fmt.Errorf("geminibatch: item %q: %w", item.ItemID, err)
			}
			var reqObj map[string]any
			if err := json.Unmarshal(body, &reqObj); err != nil {
				return nil, err
			}
			inline = append(inline, map[string]any{
				"request":  reqObj,
				"metadata": map[string]string{"key": item.ItemID},
			})
		case llm.BatchOperationEmbedding:
			emb := item.Embedding
			if emb.Model == "" {
				emb.Model = model
			}
			// One Gemini embedContent request per input string for stable item mapping.
			// We pack all inputs of one BatchItem as sequential keys itemID#i
			for i, text := range emb.Input {
				key := embItemKey(item.ItemID, i, len(emb.Input))
				inline = append(inline, map[string]any{
					"request": map[string]any{
						"content": map[string]any{
							"parts": []map[string]any{{"text": text}},
						},
					},
					"metadata": map[string]string{"key": key},
				})
			}
		}
	}
	payload := map[string]any{
		"batch": map[string]any{
			"display_name": "rho-batch",
			"input_config": map[string]any{
				"requests": map[string]any{"requests": inline},
			},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	if len(body) > maxInlineBytes {
		return nil, fmt.Errorf("geminibatch: inline batch exceeds 20MB")
	}
	digest := "sha256:" + hex.EncodeToString(sha256Bytes(body))

	path := c.baseURL + "/models/" + model + ":batchGenerateContent"
	if kind == llm.BatchOperationEmbedding {
		// Embedding batch jobs use the same batch wrapper; create endpoint is
		// model-scoped async embed (see Gemini Batch embeddings docs).
		path = c.baseURL + "/models/" + model + ":asyncBatchEmbedContent"
	}

	resp, err := c.doJSON(ctx, http.MethodPost, path, body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, llm.ErrorFromResponse("gemini", resp, c.cfg)
	}
	var wire batchResource
	if err := llm.DecodeJSONResponse(resp, c.cfg, &wire); err != nil {
		return nil, fmt.Errorf("geminibatch: decode create: %w", err)
	}
	handle, err := c.toHandle(&wire, kind, model)
	if err != nil {
		return nil, err
	}
	if opts.RecoveryKey != "" {
		handle.RecoveryKey = opts.RecoveryKey
		handle.RequestDigest = digest
	}
	if len(opts.Metadata) > 0 {
		handle.Metadata = copyMeta(opts.Metadata)
	}
	return handle, nil
}

// Recover is unsupported (no recovery-key search on Gemini Batch).
func (c *Client) Recover(ctx context.Context, recoveryKey string) (*llm.BatchHandle, error) {
	if err := llm.ValidateBatchRecoveryKey(recoveryKey); err != nil {
		return nil, err
	}
	return nil, llm.ErrBatchNotFound
}

// Get polls batch status.
func (c *Client) Get(ctx context.Context, handle llm.BatchHandle) (*llm.BatchHandle, error) {
	state, err := decodeState(handle, c.providerName)
	if err != nil {
		return nil, err
	}
	name := state.Name
	if name == "" {
		name = handle.ID
	}
	if !strings.HasPrefix(name, "batches/") && !strings.Contains(name, "/") {
		name = "batches/" + name
	}
	resp, err := c.doJSON(ctx, http.MethodGet, c.baseURL+"/"+strings.TrimPrefix(name, "/"), nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, llm.ErrorFromResponse("gemini", resp, c.cfg)
	}
	var wire batchResource
	if err := llm.DecodeJSONResponse(resp, c.cfg, &wire); err != nil {
		return nil, err
	}
	out, err := c.toHandle(&wire, handle.Operation, state.Model)
	if err != nil {
		return nil, err
	}
	out.RecoveryKey = handle.RecoveryKey
	out.RequestDigest = handle.RequestDigest
	if handle.Metadata != nil {
		out.Metadata = copyMeta(handle.Metadata)
	}
	return out, nil
}

// Results returns inline responses from a completed batch.
func (c *Client) Results(ctx context.Context, handle llm.BatchHandle) ([]llm.BatchResult, error) {
	state, err := decodeState(handle, c.providerName)
	if err != nil {
		return nil, err
	}
	// Refresh to load dest
	fresh, err := c.Get(ctx, handle)
	if err != nil {
		return nil, err
	}
	state, err = decodeState(*fresh, c.providerName)
	if err != nil {
		return nil, err
	}
	if len(state.InlinedResponses) == 0 && len(state.InlinedEmbedResponses) == 0 {
		return nil, nil
	}
	switch handle.Operation {
	case llm.BatchOperationCompletion:
		return c.parseCompletionResults(state.InlinedResponses, state.Model)
	case llm.BatchOperationEmbedding:
		return c.parseEmbeddingResults(state.InlinedEmbedResponses)
	default:
		return nil, fmt.Errorf("geminibatch: unknown operation")
	}
}

// Cancel cancels a running batch.
func (c *Client) Cancel(ctx context.Context, handle llm.BatchHandle) (*llm.BatchHandle, error) {
	state, err := decodeState(handle, c.providerName)
	if err != nil {
		return nil, err
	}
	name := state.Name
	if name == "" {
		name = handle.ID
	}
	if !strings.HasPrefix(name, "batches/") && !strings.Contains(name, "/") {
		name = "batches/" + name
	}
	resp, err := c.doJSON(ctx, http.MethodPost, c.baseURL+"/"+strings.TrimPrefix(name, "/")+":cancel", []byte("{}"))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, llm.ErrorFromResponse("gemini", resp, c.cfg)
	}
	// Cancel may return empty; re-get.
	return c.Get(ctx, handle)
}

type batchResource struct {
	Name     string `json:"name"`
	Metadata struct {
		State string `json:"state"`
	} `json:"metadata"`
	State string `json:"state"`
	Done  bool   `json:"done"`
	// dest may appear as response or dest depending on API revision
	Dest *struct {
		InlinedResponses []json.RawMessage `json:"inlinedResponses"`
		InlinedEmbeds    []json.RawMessage `json:"inlinedEmbedContentResponses"`
		FileName         string            `json:"fileName"`
	} `json:"dest"`
	Response *struct {
		InlinedResponses []json.RawMessage `json:"inlinedResponses"`
		InlinedEmbeds    []json.RawMessage `json:"inlinedEmbedContentResponses"`
	} `json:"response"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

type adapterState struct {
	Name                  string            `json:"name"`
	Model                 string            `json:"model"`
	InlinedResponses      []json.RawMessage `json:"inlined_responses,omitempty"`
	InlinedEmbedResponses []json.RawMessage `json:"inlined_embed_responses,omitempty"`
}

func (c *Client) toHandle(wire *batchResource, op llm.BatchOperationKind, model string) (*llm.BatchHandle, error) {
	if wire == nil || wire.Name == "" {
		return nil, fmt.Errorf("geminibatch: missing batch name")
	}
	stateName := wire.State
	if stateName == "" {
		stateName = wire.Metadata.State
	}
	status, err := normalizeGeminiState(stateName)
	if err != nil {
		return nil, err
	}
	st := adapterState{Name: wire.Name, Model: model}
	if wire.Dest != nil {
		st.InlinedResponses = wire.Dest.InlinedResponses
		st.InlinedEmbedResponses = wire.Dest.InlinedEmbeds
	}
	if wire.Response != nil {
		if len(st.InlinedResponses) == 0 {
			st.InlinedResponses = wire.Response.InlinedResponses
		}
		if len(st.InlinedEmbedResponses) == 0 {
			st.InlinedEmbedResponses = wire.Response.InlinedEmbeds
		}
	}
	raw, err := json.Marshal(st)
	if err != nil {
		return nil, err
	}
	id := wire.Name
	if strings.HasPrefix(id, "batches/") {
		id = strings.TrimPrefix(id, "batches/")
	}
	h := &llm.BatchHandle{
		SchemaVersion:       llm.BatchSchemaVersion,
		Provider:            c.providerName,
		ID:                  id,
		Operation:           op,
		Status:              status,
		RequestCounts:       llm.BatchRequestCounts{},
		AdapterStateVersion: adapterStateVersion,
		AdapterState:        raw,
	}
	if err := h.Validate(); err != nil {
		return nil, fmt.Errorf("geminibatch: invalid handle: %w", err)
	}
	return h, nil
}

func normalizeGeminiState(state string) (llm.BatchStatus, error) {
	// Live Gemini Batch returns BATCH_STATE_*; some docs/SDKs use JOB_STATE_*.
	switch state {
	case "JOB_STATE_PENDING", "JOB_STATE_QUEUED", "BATCH_STATE_PENDING", "BATCH_STATE_QUEUED", "":
		return llm.BatchQueued, nil
	case "JOB_STATE_RUNNING", "BATCH_STATE_RUNNING":
		return llm.BatchRunning, nil
	case "JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED":
		return llm.BatchCompleted, nil
	case "JOB_STATE_FAILED", "BATCH_STATE_FAILED":
		return llm.BatchFailed, nil
	case "JOB_STATE_CANCELLING", "BATCH_STATE_CANCELLING":
		return llm.BatchCancelling, nil
	case "JOB_STATE_CANCELLED", "BATCH_STATE_CANCELLED":
		return llm.BatchCancelled, nil
	case "JOB_STATE_EXPIRED", "BATCH_STATE_EXPIRED":
		return llm.BatchExpired, nil
	default:
		return "", fmt.Errorf("geminibatch: unknown state %q", state)
	}
}

func decodeState(handle llm.BatchHandle, provider string) (adapterState, error) {
	if err := handle.Validate(); err != nil {
		return adapterState{}, fmt.Errorf("geminibatch: invalid handle: %w", err)
	}
	if handle.Provider != provider {
		return adapterState{}, fmt.Errorf("geminibatch: provider mismatch")
	}
	if handle.AdapterStateVersion != adapterStateVersion {
		return adapterState{}, fmt.Errorf("geminibatch: unsupported adapter state version")
	}
	var st adapterState
	dec := json.NewDecoder(bytes.NewReader(handle.AdapterState))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&st); err != nil {
		return adapterState{}, err
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return adapterState{}, fmt.Errorf("geminibatch: trailing adapter state")
	}
	return st, nil
}

func (c *Client) parseCompletionResults(rows []json.RawMessage, model string) ([]llm.BatchResult, error) {
	out := make([]llm.BatchResult, 0, len(rows))
	for _, raw := range rows {
		var row struct {
			Metadata struct {
				Key string `json:"key"`
			} `json:"metadata"`
			Response json.RawMessage `json:"response"`
			Error    *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(raw, &row); err != nil {
			return nil, err
		}
		res := llm.BatchResult{ItemID: row.Metadata.Key}
		if row.Error != nil {
			res.Error = &llm.APIError{Provider: "gemini", Message: row.Error.Message}
		} else if len(row.Response) > 0 {
			msg, err := c.translator.ParseGenerateContentBody(row.Response, model)
			if err != nil {
				return nil, err
			}
			res.Response = msg
		}
		out = append(out, res)
	}
	return out, nil
}

func (c *Client) parseEmbeddingResults(rows []json.RawMessage) ([]llm.BatchResult, error) {
	// Group by base item id.
	type acc struct {
		vectors []llm.Embedding
		err     *llm.APIError
	}
	grouped := map[string]*acc{}
	order := []string{}
	for idx, raw := range rows {
		var row struct {
			Metadata struct {
				Key string `json:"key"`
			} `json:"metadata"`
			Response struct {
				Embedding struct {
					Values []float64 `json:"values"`
				} `json:"embedding"`
			} `json:"response"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(raw, &row); err != nil {
			return nil, err
		}
		base, embIndex := splitEmbKey(row.Metadata.Key, idx)
		a, ok := grouped[base]
		if !ok {
			a = &acc{}
			grouped[base] = a
			order = append(order, base)
		}
		if row.Error != nil {
			a.err = &llm.APIError{Provider: "gemini", Message: row.Error.Message}
			continue
		}
		a.vectors = append(a.vectors, llm.Embedding{Index: embIndex, Vector: row.Response.Embedding.Values})
	}
	out := make([]llm.BatchResult, 0, len(order))
	for _, id := range order {
		a := grouped[id]
		res := llm.BatchResult{ItemID: id}
		if a.err != nil {
			res.Error = a.err
		} else {
			res.Embedding = &llm.EmbeddingResponse{Embeddings: a.vectors}
		}
		out = append(out, res)
	}
	return out, nil
}

func inferKind(items []llm.BatchItem) (llm.BatchOperationKind, string, error) {
	var kind llm.BatchOperationKind
	var model string
	for _, item := range items {
		if item.Request != nil && item.Embedding != nil {
			return "", "", fmt.Errorf("geminibatch: item %q sets both request and embedding", item.ItemID)
		}
		if item.Request != nil {
			if kind == llm.BatchOperationEmbedding {
				return "", "", fmt.Errorf("geminibatch: mixed completion/embedding batch")
			}
			kind = llm.BatchOperationCompletion
			m := item.Request.Model
			if model == "" {
				model = m
			} else if m != "" && m != model {
				return "", "", fmt.Errorf("geminibatch: mixed models in batch")
			}
		} else if item.Embedding != nil {
			if kind == llm.BatchOperationCompletion {
				return "", "", fmt.Errorf("geminibatch: mixed completion/embedding batch")
			}
			kind = llm.BatchOperationEmbedding
			m := item.Embedding.Model
			if model == "" {
				model = m
			} else if m != "" && m != model {
				return "", "", fmt.Errorf("geminibatch: mixed models in batch")
			}
		}
	}
	if kind == "" || model == "" {
		return "", "", fmt.Errorf("geminibatch: model is required on every item")
	}
	return kind, model, nil
}

func embItemKey(itemID string, i, n int) string {
	if n == 1 {
		return itemID
	}
	return fmt.Sprintf("%s#%d", itemID, i)
}

func splitEmbKey(key string, fallback int) (string, int) {
	if i := strings.LastIndex(key, "#"); i > 0 {
		var idx int
		if _, err := fmt.Sscanf(key[i+1:], "%d", &idx); err == nil {
			return key[:i], idx
		}
	}
	return key, fallback
}

func (c *Client) doJSON(ctx context.Context, method, url string, body []byte) (*http.Response, error) {
	return llm.DoHTTP(ctx, c.cfg, c.httpClient, func(ctx context.Context) (*http.Request, error) {
		var rdr io.Reader
		if body != nil {
			rdr = bytes.NewReader(body)
		}
		req, err := http.NewRequestWithContext(ctx, method, url, rdr)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-goog-api-key", c.cfg.APIKey)
		return req, nil
	})
}

func copyMeta(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func sha256Bytes(b []byte) []byte {
	sum := sha256.Sum256(b)
	return sum[:]
}
