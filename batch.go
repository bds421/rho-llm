package llm

// Batch API — the asynchronous, bulk counterpart to the synchronous Client.
//
// A BatchClient submits many requests at once for offline processing (typically
// ≤24h turnaround at a reduced price) and resolves them later. It deliberately
// does NOT satisfy Client and vice-versa: Client is one-request-in/one-response-
// out; a batch is submit → poll → fetch, possibly across process restarts.
//
// The shape is provider-agnostic so additional batch backends (e.g. Anthropic
// Message Batches) can register alongside OpenAI later — see batchregister.go and
// NewBatchClient in factory.go. Transport differences (OpenAI uploads a JSONL file
// via the Files API; Anthropic submits requests inline) live entirely inside each
// driver; this file is pure neutral data + the interface.

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// BatchSchemaVersion is stamped into every serialized BatchHandle so a persisted
// handle survives refactors and a newer-than-supported version is rejected rather
// than silently mis-read (mirrors ConversationSchemaVersion).
const BatchSchemaVersion = 1

// DefaultBatchPollInterval is WaitForBatch's poll cadence when the caller passes a
// non-positive interval. Batches resolve over minutes-to-hours, so polling need not
// be aggressive.
const DefaultBatchPollInterval = 30 * time.Second

// DefaultCompletionWindow is the batch turnaround requested when BatchOptions leaves
// it blank. OpenAI currently only accepts "24h".
const DefaultCompletionWindow = "24h"

// BatchStatus is the neutral lifecycle state of a batch. Driver-specific status
// strings are mapped onto these; an unrecognized provider status passes through
// unchanged so a newly introduced state cannot crash a poll loop.
type BatchStatus string

const (
	BatchValidating BatchStatus = "validating"
	BatchInProgress BatchStatus = "in_progress"
	BatchFinalizing BatchStatus = "finalizing"
	BatchCompleted  BatchStatus = "completed"
	BatchFailed     BatchStatus = "failed"
	BatchExpired    BatchStatus = "expired"
	BatchCancelling BatchStatus = "cancelling"
	BatchCancelled  BatchStatus = "cancelled"
)

// Terminal reports whether the batch has reached a final state — no further polling
// will change it. An unknown (passed-through) status is treated as non-terminal so
// WaitForBatch keeps polling under the caller's context deadline rather than
// returning prematurely.
func (s BatchStatus) Terminal() bool {
	switch s {
	case BatchCompleted, BatchFailed, BatchExpired, BatchCancelled:
		return true
	default:
		return false
	}
}

// BatchItem is one request in a batch: a caller-assigned CustomID plus exactly one
// payload — a chat/completion Request OR an embeddings request. The CustomID is
// echoed back on every BatchResult so the (unordered) results correlate to inputs.
type BatchItem struct {
	CustomID  string            `json:"custom_id"`
	Request   *Request          `json:"request,omitempty"`
	Embedding *EmbeddingRequest `json:"embedding,omitempty"`
}

// Validate checks the per-item invariants every driver relies on: a non-empty
// CustomID and exactly one payload set. It does not decide the endpoint/kind — that
// is driver-specific (see the OpenAI driver's homogeneity check).
func (i BatchItem) Validate() error {
	if i.CustomID == "" {
		return fmt.Errorf("llm: batch item has empty custom_id")
	}
	has := 0
	if i.Request != nil {
		has++
	}
	if i.Embedding != nil {
		has++
	}
	if has != 1 {
		return fmt.Errorf("llm: batch item %q: exactly one of Request or Embedding must be set", i.CustomID)
	}
	return nil
}

// BatchResult is one resolved request, correlated by CustomID. Exactly one of
// Response / Embedding / Error is populated: a successful chat item yields Response,
// a successful embeddings item yields Embedding, and any failed line yields Error.
type BatchResult struct {
	CustomID  string             `json:"custom_id"`
	Response  *Response          `json:"response,omitempty"`
	Embedding *EmbeddingResponse `json:"embedding,omitempty"`
	Error     *APIError          `json:"error,omitempty"`
}

// BatchOptions carries optional submit-time settings.
type BatchOptions struct {
	// CompletionWindow is the requested turnaround (empty → DefaultCompletionWindow).
	CompletionWindow string
	// Metadata is attached to the batch and echoed back on the handle.
	Metadata map[string]string
}

// BatchRequestCounts is the provider's running tally of request outcomes.
type BatchRequestCounts struct {
	Total     int `json:"total"`
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
}

// BatchHandle identifies a submitted batch and carries the metadata needed to poll
// and fetch it — including across process restarts. It is a plain, versioned JSON
// value: Submit returns it, the caller persists it (MarshalJSON / LoadBatchHandle),
// and after a restart Get/Results work from the handle alone (the original items are
// gone). Endpoint is load-bearing on resume: Results selects the response parser from
// it, since the items that implied the kind no longer exist.
type BatchHandle struct {
	SchemaVersion int                `json:"schema_version"`
	Provider      string             `json:"provider"` // configured provider name (routing on resume)
	ID            string             `json:"id"`       // provider-assigned batch id
	Status        BatchStatus        `json:"status"`   // neutral lifecycle state
	Endpoint      string             `json:"endpoint"` // e.g. "/v1/chat/completions" — picks the parser
	InputFileID   string             `json:"input_file_id,omitempty"`
	OutputFileID  string             `json:"output_file_id,omitempty"`
	ErrorFileID   string             `json:"error_file_id,omitempty"`
	RequestCounts BatchRequestCounts `json:"request_counts"`
	CreatedAt     time.Time          `json:"created_at,omitempty"`
	ExpiresAt     time.Time          `json:"expires_at,omitempty"`
	Metadata      map[string]string  `json:"metadata,omitempty"`
}

// LoadBatchHandle deserializes a BatchHandle, validating the schema version. A
// missing/zero/negative version is treated as the current version (forward from
// pre-versioned writes); a version newer than this build is rejected rather than
// silently mis-read. Mirrors LoadConversation.
func LoadBatchHandle(data []byte) (*BatchHandle, error) {
	var h BatchHandle
	if err := json.Unmarshal(data, &h); err != nil {
		return nil, fmt.Errorf("llm: decode batch handle: %w", err)
	}
	if h.SchemaVersion <= 0 {
		h.SchemaVersion = BatchSchemaVersion
	}
	if h.SchemaVersion > BatchSchemaVersion {
		return nil, fmt.Errorf("llm: batch handle schema_version %d is newer than supported %d", h.SchemaVersion, BatchSchemaVersion)
	}
	return &h, nil
}

// BatchClient is the asynchronous, bulk counterpart to Client. Implementations are
// registered per protocol (RegisterBatchProvider) and constructed via NewBatchClient.
type BatchClient interface {
	// Submit uploads the items and creates the batch, returning a handle to poll.
	// It validates items up front (per-item invariants, duplicate custom_ids,
	// provider-specific homogeneity) and fails before any network effect on a bad set.
	Submit(ctx context.Context, items []BatchItem, opts BatchOptions) (*BatchHandle, error)

	// Get retrieves the current handle (status, request counts, result file ids).
	Get(ctx context.Context, id string) (*BatchHandle, error)

	// Results downloads and parses all resolved lines once the batch has output. Each
	// BatchResult is correlated by CustomID; failed lines carry Error. Calling it
	// before results exist returns whatever is available (possibly empty).
	Results(ctx context.Context, id string) ([]BatchResult, error)

	// Cancel requests cancellation and returns the updated handle.
	Cancel(ctx context.Context, id string) (*BatchHandle, error)

	// Close releases resources (idle HTTP connections).
	Close() error
}

// WaitForBatch polls Get until the batch reaches a terminal status or ctx is
// cancelled. It is caller-driven — pollInterval and ctx are the only controls, never
// a hidden long block — mirroring how Stream hands iteration control to the caller.
// On ctx cancellation it returns the last observed handle together with ctx.Err().
func WaitForBatch(ctx context.Context, bc BatchClient, id string, pollInterval time.Duration) (*BatchHandle, error) {
	if pollInterval <= 0 {
		pollInterval = DefaultBatchPollInterval
	}
	var last *BatchHandle
	for {
		h, err := bc.Get(ctx, id)
		if err != nil {
			return last, err
		}
		last = h
		if h.Status.Terminal() {
			return h, nil
		}
		select {
		case <-ctx.Done():
			return last, ctx.Err()
		case <-time.After(pollInterval):
		}
	}
}
