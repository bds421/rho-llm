// Package anthropicbatch implements Anthropic Message Batches as an llm.BatchClient.
// Transport is inline JSON (not OpenAI Files): create a batch with requests[],
// poll GET, fetch results as JSONL, cancel via POST .../cancel.
package anthropicbatch

import (
	"bufio"
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
	"github.com/bds421/rho-llm/provider/anthropic"
)

func init() {
	llm.RegisterBatchProvider("anthropic", func(cfg llm.Config) (llm.BatchClient, error) {
		return New(cfg)
	})
}

const (
	anthropicMaxTurnaround = 24 * time.Hour
	maxBatchRequests       = 10_000
	adapterStateVersion    = 1
)

// Client is the Anthropic Message Batches driver.
type Client struct {
	cfg          llm.Config
	httpClient   *http.Client
	baseURL      string
	providerName string
	translator   *anthropic.Client
}

var _ llm.BatchClient = (*Client)(nil)

// New creates an Anthropic batch client. It performs no network I/O.
func New(cfg llm.Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("anthropicbatch: API key is required")
	}
	base := llm.ResolveBaseURL(cfg)
	if base == "" {
		base = "https://api.anthropic.com/v1"
	}
	providerName := cfg.ProviderName
	if providerName == "" {
		providerName = cfg.Provider
		if providerName == "" {
			providerName = "anthropic"
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
	translator, err := anthropic.BatchTranslator(cfg)
	if err != nil {
		return nil, err
	}
	return &Client{
		cfg:          cfg,
		httpClient:   httpClient,
		baseURL:      strings.TrimRight(base, "/"),
		providerName: providerName,
		translator:   translator,
	}, nil
}

// Close releases idle HTTP connections.
func (c *Client) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}

// Submit creates a Message Batch from completion items only.
func (c *Client) Submit(ctx context.Context, items []llm.BatchItem, opts llm.BatchOptions) (*llm.BatchHandle, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	if opts.MaxTurnaround != anthropicMaxTurnaround {
		return nil, fmt.Errorf("anthropicbatch: max turnaround must be exactly 24h")
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("anthropicbatch: batch has no items")
	}
	if len(items) > maxBatchRequests {
		return nil, fmt.Errorf("anthropicbatch: batch exceeds %d requests", maxBatchRequests)
	}
	seen := make(map[string]struct{}, len(items))
	type batchReq struct {
		CustomID string          `json:"custom_id"`
		Params   json.RawMessage `json:"params"`
	}
	requests := make([]batchReq, 0, len(items))
	digestParts := make([][]byte, 0, len(items))
	for _, item := range items {
		if err := item.Validate(); err != nil {
			return nil, err
		}
		if item.Embedding != nil {
			return nil, fmt.Errorf("anthropicbatch: embeddings are not supported")
		}
		if item.Request == nil {
			return nil, fmt.Errorf("anthropicbatch: item %q missing request", item.ItemID)
		}
		if _, dup := seen[item.ItemID]; dup {
			return nil, fmt.Errorf("anthropicbatch: duplicate item_id %q", item.ItemID)
		}
		seen[item.ItemID] = struct{}{}
		params, err := c.translator.BuildMessageBatchParams(*item.Request)
		if err != nil {
			return nil, fmt.Errorf("anthropicbatch: item %q: %w", item.ItemID, err)
		}
		requests = append(requests, batchReq{CustomID: item.ItemID, Params: params})
		digestParts = append(digestParts, append([]byte(item.ItemID+"\n"), params...))
	}
	body, err := json.Marshal(map[string]any{"requests": requests})
	if err != nil {
		return nil, err
	}
	digest := "sha256:" + hex.EncodeToString(sha256Sum(body))

	resp, err := c.doJSON(ctx, http.MethodPost, c.baseURL+"/messages/batches", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, llm.ErrorFromResponse("anthropic", resp, c.cfg)
	}
	var wire batchObject
	if err := llm.DecodeJSONResponse(resp, c.cfg, &wire); err != nil {
		return nil, fmt.Errorf("anthropicbatch: decode create response: %w", err)
	}
	handle, err := c.toHandle(&wire)
	if err != nil {
		return nil, err
	}
	if opts.RecoveryKey != "" {
		handle.RecoveryKey = opts.RecoveryKey
		handle.RequestDigest = digest
	}
	if len(opts.Metadata) > 0 {
		handle.Metadata = copyMetadata(opts.Metadata)
	}
	if err := handle.Validate(); err != nil {
		return nil, fmt.Errorf("anthropicbatch: invalid handle after create: %w", err)
	}
	return handle, nil
}

// Recover is not supported: Anthropic Message Batches do not expose a
// recovery-key metadata search surface comparable to OpenAI.
func (c *Client) Recover(ctx context.Context, recoveryKey string) (*llm.BatchHandle, error) {
	if err := llm.ValidateBatchRecoveryKey(recoveryKey); err != nil {
		return nil, err
	}
	return nil, llm.ErrBatchNotFound
}

// Get refreshes a handle from the provider.
func (c *Client) Get(ctx context.Context, handle llm.BatchHandle) (*llm.BatchHandle, error) {
	if _, err := decodeAdapterState(handle, c.providerName); err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodGet, c.baseURL+"/messages/batches/"+handle.ID, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, llm.ErrorFromResponse("anthropic", resp, c.cfg)
	}
	var wire batchObject
	if err := llm.DecodeJSONResponse(resp, c.cfg, &wire); err != nil {
		return nil, err
	}
	out, err := c.toHandle(&wire)
	if err != nil {
		return nil, err
	}
	// Preserve caller-owned recovery fields not returned by Anthropic.
	out.RecoveryKey = handle.RecoveryKey
	out.RequestDigest = handle.RequestDigest
	if handle.Metadata != nil {
		out.Metadata = copyMetadata(handle.Metadata)
	}
	return out, nil
}

// Results downloads the JSONL result file once available.
func (c *Client) Results(ctx context.Context, handle llm.BatchHandle) ([]llm.BatchResult, error) {
	if _, err := decodeAdapterState(handle, c.providerName); err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodGet, c.baseURL+"/messages/batches/"+handle.ID+"/results", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, llm.ErrorFromResponse("anthropic", resp, c.cfg)
	}
	return c.parseResults(resp.Body)
}

// Cancel requests cancellation of an in-flight batch.
func (c *Client) Cancel(ctx context.Context, handle llm.BatchHandle) (*llm.BatchHandle, error) {
	if _, err := decodeAdapterState(handle, c.providerName); err != nil {
		return nil, err
	}
	resp, err := c.doJSON(ctx, http.MethodPost, c.baseURL+"/messages/batches/"+handle.ID+"/cancel", []byte("{}"))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, llm.ErrorFromResponse("anthropic", resp, c.cfg)
	}
	var wire batchObject
	if err := llm.DecodeJSONResponse(resp, c.cfg, &wire); err != nil {
		return nil, err
	}
	out, err := c.toHandle(&wire)
	if err != nil {
		return nil, err
	}
	out.RecoveryKey = handle.RecoveryKey
	out.RequestDigest = handle.RequestDigest
	if handle.Metadata != nil {
		out.Metadata = copyMetadata(handle.Metadata)
	}
	return out, nil
}

type batchObject struct {
	ID               string `json:"id"`
	Type             string `json:"type"`
	ProcessingStatus string `json:"processing_status"`
	RequestCounts    struct {
		Processing int `json:"processing"`
		Succeeded  int `json:"succeeded"`
		Errored    int `json:"errored"`
		Canceled   int `json:"canceled"`
		Expired    int `json:"expired"`
	} `json:"request_counts"`
	CreatedAt  string `json:"created_at"`
	ExpiresAt  string `json:"expires_at"`
	EndedAt    string `json:"ended_at"`
	ResultsURL string `json:"results_url"`
}

type adapterState struct {
	ResultsURL string `json:"results_url,omitempty"`
}

func (c *Client) toHandle(o *batchObject) (*llm.BatchHandle, error) {
	if o == nil || o.ID == "" {
		return nil, fmt.Errorf("anthropicbatch: missing batch id")
	}
	status, err := normalizeStatus(o.ProcessingStatus, o.RequestCounts.Canceled, o.RequestCounts.Expired, o.RequestCounts.Errored, o.RequestCounts.Succeeded)
	if err != nil {
		return nil, err
	}
	state, err := json.Marshal(adapterState{ResultsURL: o.ResultsURL})
	if err != nil {
		return nil, err
	}
	h := &llm.BatchHandle{
		SchemaVersion:       llm.BatchSchemaVersion,
		Provider:            c.providerName,
		ID:                  o.ID,
		Operation:           llm.BatchOperationCompletion,
		Status:              status,
		RequestCounts:       llm.BatchRequestCounts{Total: o.RequestCounts.Processing + o.RequestCounts.Succeeded + o.RequestCounts.Errored + o.RequestCounts.Canceled + o.RequestCounts.Expired, Completed: o.RequestCounts.Succeeded, Failed: o.RequestCounts.Errored + o.RequestCounts.Canceled + o.RequestCounts.Expired},
		AdapterStateVersion: adapterStateVersion,
		AdapterState:        state,
	}
	if t, ok := parseRFC3339(o.CreatedAt); ok {
		h.CreatedAt = t
	}
	if t, ok := parseRFC3339(o.ExpiresAt); ok {
		h.ExpiresAt = t
	}
	if err := h.Validate(); err != nil {
		return nil, fmt.Errorf("anthropicbatch: invalid provider batch handle: %w", err)
	}
	return h, nil
}

func normalizeStatus(processing string, canceled, expired, errored, succeeded int) (llm.BatchStatus, error) {
	switch processing {
	case "validating":
		return llm.BatchQueued, nil
	case "in_progress":
		return llm.BatchRunning, nil
	case "canceling":
		return llm.BatchCancelling, nil
	case "ended":
		if canceled > 0 && succeeded == 0 && errored == 0 {
			return llm.BatchCancelled, nil
		}
		if expired > 0 && succeeded == 0 && errored == 0 && canceled == 0 {
			return llm.BatchExpired, nil
		}
		if errored > 0 && succeeded == 0 && canceled == 0 && expired == 0 {
			return llm.BatchFailed, nil
		}
		return llm.BatchCompleted, nil
	default:
		return "", fmt.Errorf("anthropicbatch: unknown processing_status %q", processing)
	}
}

func decodeAdapterState(handle llm.BatchHandle, provider string) (adapterState, error) {
	if err := handle.Validate(); err != nil {
		return adapterState{}, fmt.Errorf("anthropicbatch: invalid batch handle: %w", err)
	}
	if handle.Provider != provider {
		return adapterState{}, fmt.Errorf("anthropicbatch: batch handle provider mismatch")
	}
	if handle.AdapterStateVersion != adapterStateVersion {
		return adapterState{}, fmt.Errorf("anthropicbatch: unsupported adapter state version %d", handle.AdapterStateVersion)
	}
	if handle.Operation != llm.BatchOperationCompletion {
		return adapterState{}, fmt.Errorf("anthropicbatch: operation mismatch")
	}
	var state adapterState
	decoder := json.NewDecoder(bytes.NewReader(handle.AdapterState))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&state); err != nil {
		return adapterState{}, fmt.Errorf("anthropicbatch: decode adapter state: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return adapterState{}, fmt.Errorf("anthropicbatch: decode adapter state: trailing data")
	}
	return state, nil
}

type resultLine struct {
	CustomID string `json:"custom_id"`
	Result   struct {
		Type    string          `json:"type"`
		Message json.RawMessage `json:"message"`
		Error   *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	} `json:"result"`
}

func (c *Client) parseResults(r io.Reader) ([]llm.BatchResult, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(nil, c.cfg.EffectiveMaxSSELineBytes())
	var out []llm.BatchResult
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var row resultLine
		if err := json.Unmarshal(line, &row); err != nil {
			return nil, fmt.Errorf("anthropicbatch: decode result line: %w", err)
		}
		res := llm.BatchResult{ItemID: row.CustomID}
		switch row.Result.Type {
		case "succeeded":
			msg, err := c.translator.ParseMessageBatchResult(row.Result.Message)
			if err != nil {
				return nil, err
			}
			res.Response = msg
		case "errored", "canceled", "expired":
			msg := "batch item failed"
			if row.Result.Error != nil && row.Result.Error.Message != "" {
				msg = row.Result.Error.Message
			}
			if row.Result.Type != "" {
				msg = row.Result.Type + ": " + msg
			}
			res.Error = &llm.APIError{Provider: "anthropic", StatusCode: 0, Message: msg}
		default:
			return nil, fmt.Errorf("anthropicbatch: unknown result type %q", row.Result.Type)
		}
		out = append(out, res)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("x-api-key", c.cfg.APIKey)
	req.Header.Set("anthropic-version", c.cfg.EffectiveAnthropicVersion())
	req.Header.Set("Content-Type", "application/json")
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
		c.setHeaders(req)
		return req, nil
	})
}

func parseRFC3339(s string) (time.Time, bool) {
	if s == "" {
		return time.Time{}, false
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Time{}, false
	}
	return t.UTC(), true
}

func copyMetadata(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func sha256Sum(b []byte) []byte {
	sum := sha256.Sum256(b)
	return sum[:]
}
