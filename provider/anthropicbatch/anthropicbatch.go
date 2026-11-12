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
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"regexp"
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
	budgetThinking := false
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
		if err := llm.ValidateRequestCapabilities(c.cfg, *item.Request, false); err != nil {
			return nil, fmt.Errorf("anthropicbatch: item %q: %w", item.ItemID, err)
		}
		if err := llm.RequireCapabilitiesForModel(c.cfg, item.Request.Model, llm.CapabilityBatch); err != nil {
			return nil, fmt.Errorf("anthropicbatch: item %q: %w", item.ItemID, err)
		}
		params, err := c.translator.BuildMessageBatchParams(*item.Request)
		if err != nil {
			return nil, fmt.Errorf("anthropicbatch: item %q: %w", item.ItemID, err)
		}
		budgetThinking = budgetThinking || usesBudgetThinking(params)
		requests = append(requests, batchReq{CustomID: item.ItemID, Params: params})
		digestParts = append(digestParts, append([]byte(item.ItemID+"\n"), params...))
	}
	body, err := json.Marshal(map[string]any{"requests": requests})
	if err != nil {
		return nil, err
	}
	digest := "sha256:" + hex.EncodeToString(sha256Sum(body))

	resp, err := c.doJSONWithOptions(ctx, http.MethodPost, c.baseURL+"/messages/batches", body, anthropic.BetaFeatures(c.cfg.BetaFeatures, budgetThinking), llm.HTTPCallOptions{NonIdempotent: true})
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
	resp, err := c.doJSON(ctx, http.MethodGet, c.batchURL(handle.ID, ""), nil)
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
	resp, err := c.doJSON(ctx, http.MethodGet, c.batchURL(handle.ID, "/results"), nil)
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
	resp, err := c.doJSON(ctx, http.MethodPost, c.batchURL(handle.ID, "/cancel"), []byte("{}"))
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
	if err := checkBatchID(handle.ID); err != nil {
		return adapterState{}, err
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

// batchIDPattern is the shape of an Anthropic batch ID (msgbatch_…). A handle
// is often loaded back from storage, so its ID is checked before it becomes
// part of a URL path: "../" or "?" would otherwise reach another endpoint
// with the deployment's key (B29).
var batchIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,128}$`)

// batchURL returns the URL of a batch, or of one of its sub-resources. The ID
// was checked by decodeAdapterState (checkBatchID).
func (c *Client) batchURL(id, suffix string) string {
	return c.baseURL + "/messages/batches/" + url.PathEscape(id) + suffix
}

// checkBatchID rejects a batch ID that is not an Anthropic batch ID.
func checkBatchID(id string) error {
	if !batchIDPattern.MatchString(id) {
		return fmt.Errorf("anthropicbatch: invalid batch id %q", id)
	}
	return nil
}

type resultLine struct {
	CustomID string `json:"custom_id"`
	Result   struct {
		Type    string          `json:"type"`
		Message json.RawMessage `json:"message"`
		Error   *resultError    `json:"error"`
	} `json:"result"`
}

// resultError is an errored item's error. The API nests the error object in
// an envelope ({"type":"error","error":{"type":…,"message":…}}); the flat
// shape is accepted too. Before B27 only the flat shape was read, so every
// errored item reported "batch item failed" without the provider's message.
type resultError struct {
	Type    string       `json:"type"`
	Message string       `json:"message"`
	Error   *resultError `json:"error"`
}

// detail returns the innermost error's type and message.
func (e *resultError) detail() (errType, message string) {
	for e.Error != nil && (e.Message == "" || e.Type == "error") {
		e = e.Error
	}
	return e.Type, e.Message
}

// limitedBody counts what a result download reads, so the whole file can be
// held to Config.MaxBatchDownloadBytes.
type limitedBody struct {
	r     io.Reader
	n     int64
	limit int64
}

func (l *limitedBody) Read(p []byte) (int, error) {
	if l.n > l.limit {
		return 0, errResultsTooLarge
	}
	if rest := l.limit + 1 - l.n; int64(len(p)) > rest {
		p = p[:rest]
	}
	n, err := l.r.Read(p)
	l.n += int64(n)
	if l.n > l.limit {
		return n, errResultsTooLarge
	}
	return n, err
}

var errResultsTooLarge = errors.New("anthropicbatch: batch result file exceeds Config.MaxBatchDownloadBytes (raise it)")

// parseResults reads the JSONL result file. One line holds a whole response,
// so a line may be as large as the file: it is bounded by
// Config.MaxBatchDownloadBytes like the download itself, not by the 256 KB SSE
// line cap, which failed any batch with a long answer (B26).
func (c *Client) parseResults(r io.Reader) ([]llm.BatchResult, error) {
	limit := c.cfg.EffectiveMaxBatchDownloadBytes()
	body := &limitedBody{r: r, limit: limit}
	scanner := bufio.NewScanner(body)
	scanner.Buffer(nil, int(min(limit+1, int64(math.MaxInt32))))
	var out []llm.BatchResult
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var row resultLine
		if err := json.Unmarshal(line, &row); err != nil {
			if body.n > limit {
				return nil, errResultsTooLarge // the cap cut this line short
			}
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
			if row.Result.Error != nil {
				errType, message := row.Result.Error.detail()
				if message != "" {
					msg = message
				}
				if errType != "" && errType != "error" {
					msg = errType + ": " + msg
				}
			}
			msg = row.Result.Type + ": " + msg
			res.Error = &llm.APIError{Provider: "anthropic", StatusCode: 0, Message: msg}
		default:
			return nil, fmt.Errorf("anthropicbatch: unknown result type %q", row.Result.Type)
		}
		out = append(out, res)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if body.n > limit {
		return nil, errResultsTooLarge
	}
	return out, nil
}

func (c *Client) setHeaders(req *http.Request, betas []string) {
	req.Header.Set("x-api-key", c.cfg.APIKey)
	req.Header.Set("anthropic-version", c.cfg.EffectiveAnthropicVersion())
	req.Header.Set("Content-Type", "application/json")
	if len(betas) > 0 {
		req.Header.Set("anthropic-beta", strings.Join(betas, ","))
	}
}

func (c *Client) doJSON(ctx context.Context, method, url string, body []byte) (*http.Response, error) {
	return c.doJSONWithOptions(ctx, method, url, body, nil, llm.HTTPCallOptions{})
}

// doJSONWithOptions lets batch creation mark its POST non-idempotent (H5):
// a resend after a 5xx the provider already committed would submit — and
// bill — a duplicate batch.
func (c *Client) doJSONWithOptions(ctx context.Context, method, url string, body []byte, betas []string, opts llm.HTTPCallOptions) (*http.Response, error) {
	return llm.DoHTTPWithOptions(ctx, c.cfg, c.httpClient, func(ctx context.Context) (*http.Request, error) {
		var rdr io.Reader
		if body != nil {
			rdr = bytes.NewReader(body)
		}
		req, err := http.NewRequestWithContext(ctx, method, url, rdr)
		if err != nil {
			return nil, err
		}
		c.setHeaders(req, betas)
		return req, nil
	}, opts)
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

// usesBudgetThinking reports whether built params enable budget thinking, the
// one case the interleaved-thinking beta is sent for.
func usesBudgetThinking(params json.RawMessage) bool {
	var p struct {
		Thinking *struct {
			Type string `json:"type"`
		} `json:"thinking"`
	}
	return json.Unmarshal(params, &p) == nil && p.Thinking != nil && p.Thinking.Type == "enabled"
}
