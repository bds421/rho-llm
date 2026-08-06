package openaibatch

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"strings"
	"time"

	"github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

const (
	recoveryMetadataKey  = "rho_recovery_key"
	requestDigestMetaKey = "rho_request_digest"
	recoveryPageSize     = 100
	maximumRecoveryPages = 10
)

// batchObject is the OpenAI Batch resource as returned by create/get/cancel.
type batchObject struct {
	ID            string `json:"id"`
	Endpoint      string `json:"endpoint"`
	Status        string `json:"status"`
	InputFileID   string `json:"input_file_id"`
	OutputFileID  string `json:"output_file_id"`
	ErrorFileID   string `json:"error_file_id"`
	CreatedAt     int64  `json:"created_at"`
	ExpiresAt     int64  `json:"expires_at"`
	RequestCounts struct {
		Total     int `json:"total"`
		Completed int `json:"completed"`
		Failed    int `json:"failed"`
	} `json:"request_counts"`
	Metadata map[string]string `json:"metadata"`
	// Errors carries batch-level (not per-request) validation failures.
	Errors *struct {
		Data []struct {
			Code    string `json:"code"`
			Message string `json:"message"`
			Line    *int   `json:"line"`
		} `json:"data"`
	} `json:"errors"`
}

// toHandle maps the wire object to the neutral, serializable BatchHandle. OpenAI
// lifecycle states are normalized into the closed root lifecycle before exposure.
func (c *Client) toHandle(o *batchObject) (*llm.BatchHandle, error) {
	metadata := make(map[string]string, len(o.Metadata))
	for key, value := range o.Metadata {
		if key != recoveryMetadataKey && key != requestDigestMetaKey {
			metadata[key] = value
		}
	}
	if len(metadata) == 0 {
		metadata = nil
	}
	status, err := normalizeStatus(o.Status)
	if err != nil {
		return nil, err
	}
	state, operation, err := encodeAdapterState(o)
	if err != nil {
		return nil, err
	}
	h := &llm.BatchHandle{
		SchemaVersion: llm.BatchSchemaVersion,
		Provider:      c.providerName,
		ID:            o.ID,
		Operation:     operation,
		Status:        status,
		RequestCounts: llm.BatchRequestCounts{
			Total:     o.RequestCounts.Total,
			Completed: o.RequestCounts.Completed,
			Failed:    o.RequestCounts.Failed,
		},
		Metadata:            metadata,
		RecoveryKey:         o.Metadata[recoveryMetadataKey],
		RequestDigest:       o.Metadata[requestDigestMetaKey],
		AdapterStateVersion: adapterStateVersion,
		AdapterState:        state,
	}
	if o.CreatedAt > 0 {
		h.CreatedAt = time.Unix(o.CreatedAt, 0).UTC()
	}
	if o.ExpiresAt > 0 {
		h.ExpiresAt = time.Unix(o.ExpiresAt, 0).UTC()
	}
	if err := h.Validate(); err != nil {
		return nil, fmt.Errorf("openaibatch: invalid provider batch handle: %w", err)
	}
	return h, nil
}

func normalizeStatus(status string) (llm.BatchStatus, error) {
	switch status {
	case "validating":
		return llm.BatchQueued, nil
	case "in_progress", "finalizing":
		return llm.BatchRunning, nil
	case "completed":
		return llm.BatchCompleted, nil
	case "failed":
		return llm.BatchFailed, nil
	case "expired":
		return llm.BatchExpired, nil
	case "cancelling":
		return llm.BatchCancelling, nil
	case "cancelled":
		return llm.BatchCancelled, nil
	default:
		return "", fmt.Errorf("openaibatch: unknown provider batch status %q", status)
	}
}

// getObject retrieves the raw batch resource (used by Get and Results).
func (c *Client) getObject(ctx context.Context, handle llm.BatchHandle) (*batchObject, error) {
	state, err := decodeAdapterState(handle, c.providerName)
	if err != nil {
		return nil, err
	}
	var o batchObject
	if err := c.doJSON(ctx, "GET", c.baseURL+"/batches/"+url.PathEscape(handle.ID), nil, &o); err != nil {
		return nil, err
	}
	refreshed, err := c.toHandle(&o)
	if err != nil {
		return nil, err
	}
	if refreshed.ID != handle.ID || refreshed.Provider != handle.Provider ||
		refreshed.Operation != handle.Operation {
		return nil, fmt.Errorf("openaibatch: provider batch identity drift")
	}
	refreshedState, err := decodeAdapterState(*refreshed, c.providerName)
	if err != nil {
		return nil, err
	}
	if refreshedState.Endpoint != state.Endpoint || refreshedState.InputFileID != state.InputFileID {
		return nil, fmt.Errorf("openaibatch: provider batch resume state drift")
	}
	return &o, nil
}

// createBatch creates a batch referencing the uploaded input file.
func (c *Client) createBatch(ctx context.Context, inputFileID, endpoint, window string, metadata map[string]string) (*llm.BatchHandle, error) {
	body := map[string]any{
		"input_file_id":     inputFileID,
		"endpoint":          endpoint,
		"completion_window": window,
	}
	if len(metadata) > 0 {
		body["metadata"] = metadata
	}
	var o batchObject
	if err := c.doJSONSingle(ctx, "POST", c.baseURL+"/batches", body, &o); err != nil {
		return nil, err
	}
	return c.toHandle(&o)
}

// Get retrieves the current batch handle.
func (c *Client) Get(ctx context.Context, handle llm.BatchHandle) (*llm.BatchHandle, error) {
	o, err := c.getObject(ctx, handle)
	if err != nil {
		return nil, err
	}
	return c.toHandle(o)
}

// Recover scans a bounded provider inventory for one exact recovery key. It scans
// through the last page even after finding a match so duplicates can never be
// silently accepted as successful reconciliation.
func (c *Client) Recover(ctx context.Context, recoveryKey string) (*llm.BatchHandle, error) {
	if err := llm.ValidateBatchRecoveryKey(recoveryKey); err != nil {
		return nil, err
	}
	var found *llm.BatchHandle
	after := ""
	for page := 0; page < maximumRecoveryPages; page++ {
		endpoint := fmt.Sprintf("%s/batches?limit=%d", c.baseURL, recoveryPageSize)
		if after != "" {
			endpoint += "&after=" + url.QueryEscape(after)
		}
		var inventory struct {
			Data    []batchObject `json:"data"`
			HasMore bool          `json:"has_more"`
			LastID  string        `json:"last_id"`
		}
		if err := c.doJSON(ctx, "GET", endpoint, nil, &inventory); err != nil {
			return nil, err
		}
		for index := range inventory.Data {
			if inventory.Data[index].Metadata[recoveryMetadataKey] != recoveryKey {
				continue
			}
			handle, err := c.toHandle(&inventory.Data[index])
			if err != nil {
				return nil, err
			}
			if found != nil && found.ID != handle.ID {
				return nil, fmt.Errorf("%w: %q", llm.ErrBatchRecoveryConflict, recoveryKey)
			}
			found = handle
		}
		if !inventory.HasMore {
			if found == nil {
				return nil, fmt.Errorf("%w: %q", llm.ErrBatchNotFound, recoveryKey)
			}
			return found, nil
		}
		next := strings.TrimSpace(inventory.LastID)
		if next == "" || next == after {
			return nil, fmt.Errorf("openaibatch: invalid recovery pagination cursor")
		}
		after = next
	}
	return nil, fmt.Errorf("openaibatch: recovery inventory exceeds %d pages", maximumRecoveryPages)
}

// Cancel requests cancellation and reconciles an ambiguous response with a safe
// GET. The POST itself is always single-dispatch; durable callers own retries.
func (c *Client) Cancel(ctx context.Context, handle llm.BatchHandle) (*llm.BatchHandle, error) {
	originalState, err := decodeAdapterState(handle, c.providerName)
	if err != nil {
		return nil, err
	}
	current, err := c.Get(ctx, handle)
	if err != nil {
		return nil, err
	}
	if current.Status.Terminal() || current.Status == llm.BatchCancelling {
		return current, nil
	}
	var o batchObject
	if err = c.doJSONSingle(ctx, "POST", c.baseURL+"/batches/"+url.PathEscape(handle.ID)+"/cancel", nil, &o); err != nil {
		reconciled, getErr := c.Get(ctx, handle)
		if getErr == nil && (reconciled.Status.Terminal() ||
			reconciled.Status == llm.BatchCancelling) {
			return reconciled, nil
		}
		return nil, errors.Join(err, getErr)
	}
	updated, err := c.toHandle(&o)
	if err != nil {
		return nil, err
	}
	if updated.ID != handle.ID || updated.Provider != handle.Provider ||
		updated.Operation != handle.Operation {
		return nil, fmt.Errorf("openaibatch: provider batch identity drift")
	}
	updatedState, err := decodeAdapterState(*updated, c.providerName)
	if err != nil {
		return nil, err
	}
	if updatedState.Endpoint != originalState.Endpoint ||
		updatedState.InputFileID != originalState.InputFileID {
		return nil, fmt.Errorf("openaibatch: provider batch resume state drift")
	}
	if updated.Status.Terminal() || updated.Status == llm.BatchCancelling {
		return updated, nil
	}
	return nil, fmt.Errorf("openaibatch: cancel returned non-cancelling status %q", updated.Status)
}

// outputLine is one line of a batch output/error file.
type outputLine struct {
	CustomID string `json:"custom_id"`
	Response *struct {
		StatusCode int             `json:"status_code"`
		RequestID  string          `json:"request_id"`
		Body       json.RawMessage `json:"body"`
	} `json:"response"`
	Error json.RawMessage `json:"error"`
}

// Results downloads and parses all resolved lines, correlated by ItemID. The parser
// is selected from adapter-owned endpoint state (the original items are gone on resume).
// Output and error files are merged; the first occurrence of a custom_id wins. A
// malformed line yields an error result rather than aborting the whole download. When
// the batch has no result files but carries batch-level errors, those are surfaced.
func (c *Client) Results(ctx context.Context, handle llm.BatchHandle) ([]llm.BatchResult, error) {
	o, err := c.getObject(ctx, handle)
	if err != nil {
		return nil, err
	}

	// Build the codec once, from the endpoint — items no longer exist on resume.
	var chatT *openaicompat.Client
	var respT *openairesponses.Client
	switch o.Endpoint {
	case endpointChat:
		if chatT, err = openaicompat.BatchTranslator(c.cfg); err != nil {
			return nil, err
		}
	case endpointResponses:
		if respT, err = openairesponses.BatchTranslator(c.cfg); err != nil {
			return nil, err
		}
	}

	var results []llm.BatchResult
	seen := make(map[string]struct{})

	for _, fileID := range []string{o.OutputFileID, o.ErrorFileID} {
		if fileID == "" {
			continue
		}
		data, err := c.downloadFile(ctx, fileID)
		if err != nil {
			return nil, err
		}
		if err := forEachLine(data, func(raw []byte) {
			res, ok := c.parseOutputLine(o.Endpoint, raw, chatT, respT)
			if !ok {
				return
			}
			if res.ItemID != "" {
				if _, dup := seen[res.ItemID]; dup {
					return
				}
				seen[res.ItemID] = struct{}{}
			}
			results = append(results, res)
		}); err != nil {
			return nil, err
		}
	}

	// Batch-level validation errors (no per-request result files).
	if len(results) == 0 && o.Errors != nil {
		for _, e := range o.Errors.Data {
			results = append(results, llm.BatchResult{
				Error: llm.NewAPIErrorFromStatusWithLimit(c.providerName, 0, e.Message, c.cfg.EffectiveMaxErrorMessageLen()),
			})
		}
	}
	return results, nil
}

// parseOutputLine turns one JSONL line into a BatchResult. ok is false only for blank
// lines. A line that cannot be parsed, carries an error, or has a non-2xx status_code
// becomes an error result; a 2xx line is parsed into a Response/Embedding by endpoint.
func (c *Client) parseOutputLine(endpoint string, raw []byte, chatT *openaicompat.Client, respT *openairesponses.Client) (llm.BatchResult, bool) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return llm.BatchResult{}, false
	}
	var line outputLine
	if err := json.Unmarshal(raw, &line); err != nil {
		// Unparseable line — can't correlate to a custom_id. Surface it as an error
		// result rather than dropping it or failing the whole batch.
		return llm.BatchResult{Error: c.apiError(0, "malformed batch result line: "+err.Error())}, true
	}
	res := llm.BatchResult{ItemID: line.CustomID}

	if len(line.Error) > 0 && !bytes.Equal(bytes.TrimSpace(line.Error), []byte("null")) {
		res.Error = c.errorFromBody(0, line.Error)
		return res, true
	}
	if line.Response == nil {
		res.Error = c.apiError(0, "batch line had neither response nor error")
		return res, true
	}
	if line.Response.StatusCode/100 != 2 {
		res.Error = c.errorFromBody(line.Response.StatusCode, line.Response.Body)
		return res, true
	}

	switch endpoint {
	case endpointChat:
		if chatT == nil {
			res.Error = c.apiError(0, "no chat translator for endpoint")
			break
		}
		r, err := chatT.ParseChatBatchResultBody(line.Response.Body)
		if err != nil {
			res.Error = c.apiError(0, "decode chat result body: "+err.Error())
		} else {
			res.Response = r
		}
	case endpointResponses:
		if respT == nil {
			res.Error = c.apiError(0, "no responses translator for endpoint")
			break
		}
		r, err := respT.ParseResponsesBatchResultBody(line.Response.Body)
		if err != nil {
			res.Error = c.apiError(0, "decode responses result body: "+err.Error())
		} else {
			res.Response = r
		}
	case endpointEmbeddings:
		em, err := openaicompat.ParseEmbeddingsBatchResultBody(line.Response.Body, 0)
		if err != nil {
			res.Error = c.apiError(0, "decode embeddings result body: "+err.Error())
		} else {
			res.Embedding = em
		}
	default:
		res.Error = c.apiError(0, fmt.Sprintf("unknown batch endpoint %q", endpoint))
	}
	return res, true
}

// apiError builds a bounded APIError for the given status and message.
func (c *Client) apiError(status int, msg string) *llm.APIError {
	return llm.NewAPIErrorFromStatusWithLimit(c.providerName, status, msg, c.cfg.EffectiveMaxErrorMessageLen())
}

// errorFromBody extracts a human message from an error body ({"message":...} or
// {"error":{"message":...}}), falling back to the raw body.
func (c *Client) errorFromBody(status int, body json.RawMessage) *llm.APIError {
	msg := string(body)
	var probe struct {
		Message string `json:"message"`
		Error   struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &probe) == nil {
		if probe.Error.Message != "" {
			msg = probe.Error.Message
		} else if probe.Message != "" {
			msg = probe.Message
		}
	}
	return c.apiError(status, msg)
}

// forEachLine invokes fn for each newline-delimited line of data. Uses a bufio.Reader
// (not Scanner) so a single very large line is not capped.
func forEachLine(data []byte, fn func([]byte)) error {
	r := bufio.NewReader(bytes.NewReader(data))
	for {
		line, err := r.ReadBytes('\n')
		if len(line) > 0 {
			fn(line)
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}
