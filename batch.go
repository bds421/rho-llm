package llm

// Batch API — the asynchronous, bulk counterpart to the synchronous Client.
//
// A BatchClient submits many requests at once for offline processing under an
// exact caller-authored maximum turnaround and resolves them later. It deliberately
// does NOT satisfy Client and vice-versa: Client is one-request-in/one-response-
// out; a batch is submit → poll → fetch, possibly across process restarts.
//
// The shape is provider-agnostic so additional batch backends (e.g. Anthropic
// Message Batches) can register alongside OpenAI later — see batchregister.go and
// NewBatchClient in factory.go. Transport differences (OpenAI uploads a JSONL file
// via the Files API; Anthropic submits requests inline) live entirely inside each
// driver; this file is pure neutral data + the interface.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

// BatchSchemaVersion is stamped into every serialized BatchHandle. Only this
// exact schema is accepted; provider transport state is deliberately opaque.
const BatchSchemaVersion = 2

// DefaultBatchPollInterval is WaitForBatch's poll cadence when the caller passes a
// non-positive interval. Batches resolve over minutes-to-hours, so polling need not
// be aggressive.
const DefaultBatchPollInterval = 30 * time.Second

const (
	MaxBatchRecoveryKeyBytes  = 128
	MaxBatchMetadataEntries   = 16
	MaxBatchMetadataKeyBytes  = 64
	MaxBatchMetadataValBytes  = 512
	MaxBatchAdapterStateBytes = 16 << 10
	MaxBatchTurnaround        = 30 * 24 * time.Hour
	maxBatchProviderBytes     = 128
	maxBatchOpaqueIDBytes     = 256
	maxBatchAdapterVersion    = 1 << 16
)

var (
	ErrBatchNotFound         = errors.New("llm: batch recovery key not found")
	ErrBatchRecoveryConflict = errors.New("llm: batch recovery key matched multiple batches")
	batchRecoveryKeyPattern  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]*$`)
)

// ValidateBatchRecoveryKey validates the caller-owned, provider-visible identity
// used to reconcile an ambiguous Submit across process restarts. It deliberately
// accepts only bounded printable ASCII so it is safe in provider metadata and logs.
func ValidateBatchRecoveryKey(value string) error {
	if value == "" || len(value) > MaxBatchRecoveryKeyBytes ||
		!batchRecoveryKeyPattern.MatchString(value) {
		return fmt.Errorf("llm: invalid batch recovery key")
	}
	return nil
}

// BatchStatus is the closed neutral lifecycle state of a batch. Every adapter
// maps its provider states onto these values and rejects unknown provider states.
type BatchStatus string

const (
	BatchQueued     BatchStatus = "queued"
	BatchRunning    BatchStatus = "running"
	BatchCompleted  BatchStatus = "completed"
	BatchFailed     BatchStatus = "failed"
	BatchExpired    BatchStatus = "expired"
	BatchCancelling BatchStatus = "cancelling"
	BatchCancelled  BatchStatus = "cancelled"
)

func (s BatchStatus) valid() bool {
	switch s {
	case BatchQueued, BatchRunning, BatchCompleted, BatchFailed, BatchExpired,
		BatchCancelling, BatchCancelled:
		return true
	default:
		return false
	}
}

// Terminal reports whether a validated batch has reached a final state.
func (s BatchStatus) Terminal() bool {
	switch s {
	case BatchCompleted, BatchFailed, BatchExpired, BatchCancelled:
		return true
	default:
		return false
	}
}

// BatchItem is one request in a batch: a caller-assigned ItemID plus exactly one
// payload — a chat/completion Request OR an embeddings request. The ItemID is
// echoed back on every BatchResult so the (unordered) results correlate to inputs.
type BatchItem struct {
	ItemID    string            `json:"item_id"`
	Request   *Request          `json:"request,omitempty"`
	Embedding *EmbeddingRequest `json:"embedding,omitempty"`
}

// Validate checks the per-item invariants every driver relies on: a non-empty
// ItemID and exactly one payload set. It does not decide the endpoint/kind — that
// is driver-specific (see the OpenAI driver's homogeneity check).
func (i BatchItem) Validate() error {
	if i.ItemID == "" {
		return fmt.Errorf("llm: batch item has empty item_id")
	}
	has := 0
	if i.Request != nil {
		has++
	}
	if i.Embedding != nil {
		has++
	}
	if has != 1 {
		return fmt.Errorf("llm: batch item %q: exactly one of Request or Embedding must be set", i.ItemID)
	}
	return nil
}

// BatchResult is one resolved request, correlated by ItemID. Exactly one of
// Response / Embedding / Error is populated: a successful chat item yields Response,
// a successful embeddings item yields Embedding, and any failed line yields Error.
type BatchResult struct {
	ItemID    string             `json:"item_id"`
	Response  *Response          `json:"response,omitempty"`
	Embedding *EmbeddingResponse `json:"embedding,omitempty"`
	Error     *APIError          `json:"error,omitempty"`
}

// BatchOptions carries optional submit-time settings.
type BatchOptions struct {
	// MaxTurnaround is the longest acceptable provider turnaround. It is required;
	// each adapter proves whether it can represent the exact duration before I/O.
	MaxTurnaround time.Duration
	// Metadata is attached to the batch and echoed back on the handle.
	Metadata map[string]string
	// RecoveryKey is a stable caller-owned idempotency/reconciliation identity.
	// Drivers that support recovery persist it in provider metadata; callers can
	// later call Recover after an ambiguous Submit or process restart.
	RecoveryKey string
}

func (options BatchOptions) Validate() error {
	if options.MaxTurnaround <= 0 || options.MaxTurnaround > MaxBatchTurnaround {
		return fmt.Errorf("llm: invalid batch max turnaround")
	}
	if options.RecoveryKey != "" {
		if err := ValidateBatchRecoveryKey(options.RecoveryKey); err != nil {
			return err
		}
	}
	return validateBatchMetadata(options.Metadata)
}

// BatchRequestCounts is the provider's running tally of request outcomes.
type BatchRequestCounts struct {
	Total     int `json:"total"`
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
}

// BatchOperationKind is the provider-neutral semantic operation of a homogeneous
// batch. Wire endpoints and provider subtypes remain inside adapter state.
type BatchOperationKind string

const (
	BatchOperationCompletion BatchOperationKind = "completion"
	BatchOperationEmbedding  BatchOperationKind = "embedding"
)

func (kind BatchOperationKind) valid() bool {
	return kind == BatchOperationCompletion || kind == BatchOperationEmbedding
}

// BatchHandle identifies a submitted batch and carries the metadata needed to poll
// and fetch it across process restarts. AdapterState is bounded, versioned, opaque
// JSON: only the selected provider adapter may interpret remote identifiers or wire
// routing state. Callers persist and pass the complete handle back to the client.
type BatchHandle struct {
	SchemaVersion       int                `json:"schema_version"`
	Provider            string             `json:"provider"`
	ID                  string             `json:"id"`
	Operation           BatchOperationKind `json:"operation"`
	Status              BatchStatus        `json:"status"`
	RequestCounts       BatchRequestCounts `json:"request_counts"`
	CreatedAt           time.Time          `json:"created_at,omitempty"`
	ExpiresAt           time.Time          `json:"expires_at,omitempty"`
	Metadata            map[string]string  `json:"metadata,omitempty"`
	RecoveryKey         string             `json:"recovery_key,omitempty"`
	RequestDigest       string             `json:"request_digest,omitempty"`
	AdapterStateVersion int                `json:"adapter_state_version"`
	AdapterState        json.RawMessage    `json:"adapter_state"`
}

// Validate checks every neutral persisted field needed to resume polling safely.
// Provider-specific state remains opaque here and is validated by its adapter.
func (h BatchHandle) Validate() error {
	if h.SchemaVersion != BatchSchemaVersion {
		return fmt.Errorf("llm: unsupported batch handle schema_version %d", h.SchemaVersion)
	}
	if err := validateBatchOpaque("provider", h.Provider, maxBatchProviderBytes); err != nil {
		return err
	}
	if err := validateBatchOpaque("id", h.ID, maxBatchOpaqueIDBytes); err != nil {
		return err
	}
	if !h.Operation.valid() {
		return fmt.Errorf("llm: invalid batch operation")
	}
	if !h.Status.valid() {
		return fmt.Errorf("llm: invalid batch status")
	}
	if h.AdapterStateVersion <= 0 || h.AdapterStateVersion > maxBatchAdapterVersion {
		return fmt.Errorf("llm: invalid batch adapter state version")
	}
	state := bytes.TrimSpace(h.AdapterState)
	if len(state) == 0 || len(state) > MaxBatchAdapterStateBytes || state[0] != '{' || !json.Valid(state) {
		return fmt.Errorf("llm: invalid batch adapter state")
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(state, &object); err != nil || object == nil {
		return fmt.Errorf("llm: invalid batch adapter state")
	}
	if h.RequestCounts.Total < 0 || h.RequestCounts.Completed < 0 ||
		h.RequestCounts.Failed < 0 ||
		h.RequestCounts.Completed+h.RequestCounts.Failed > h.RequestCounts.Total {
		return fmt.Errorf("llm: invalid batch request counts")
	}
	for label, value := range map[string]time.Time{
		"created_at": h.CreatedAt, "expires_at": h.ExpiresAt,
	} {
		if !value.IsZero() && value.Location() != time.UTC {
			return fmt.Errorf("llm: batch handle %s must be UTC", label)
		}
	}
	if !h.CreatedAt.IsZero() && !h.ExpiresAt.IsZero() &&
		!h.ExpiresAt.After(h.CreatedAt) {
		return fmt.Errorf("llm: batch handle expires_at must be after created_at")
	}
	if h.RecoveryKey != "" {
		if err := ValidateBatchRecoveryKey(h.RecoveryKey); err != nil {
			return err
		}
		if !validBatchRequestDigest(h.RequestDigest) {
			return fmt.Errorf("llm: invalid batch request digest")
		}
	} else if h.RequestDigest != "" {
		return fmt.Errorf("llm: batch request digest requires a recovery key")
	}
	return validateBatchMetadata(h.Metadata)
}

// LoadBatchHandle deserializes and validates the exact current BatchHandle schema.
// Missing, older, and newer versions fail closed; there is no legacy decoder.
func LoadBatchHandle(data []byte) (*BatchHandle, error) {
	var h BatchHandle
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&h); err != nil {
		return nil, fmt.Errorf("llm: decode batch handle: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("llm: decode batch handle: trailing data")
	}
	if err := h.Validate(); err != nil {
		return nil, err
	}
	return &h, nil
}

func validateBatchMetadata(metadata map[string]string) error {
	if len(metadata) > MaxBatchMetadataEntries {
		return fmt.Errorf("llm: batch metadata has too many entries")
	}
	for key, value := range metadata {
		if err := validateBatchOpaque("metadata key", key, MaxBatchMetadataKeyBytes); err != nil {
			return err
		}
		if value != strings.TrimSpace(value) || len(value) > MaxBatchMetadataValBytes ||
			!utf8.ValidString(value) || hasBatchControlRune(value) {
			return fmt.Errorf("llm: invalid batch metadata value")
		}
	}
	return nil
}

func hasBatchControlRune(value string) bool {
	for _, r := range value {
		if r < 0x20 || r == 0x7f {
			return true
		}
	}
	return false
}

func validateBatchOpaque(label, value string, maximum int) error {
	if value == "" || value != strings.TrimSpace(value) || len(value) > maximum ||
		!utf8.ValidString(value) {
		return fmt.Errorf("llm: invalid batch %s", label)
	}
	for _, r := range value {
		if r < 0x21 || r > 0x7e {
			return fmt.Errorf("llm: invalid batch %s", label)
		}
	}
	return nil
}

func validBatchRequestDigest(value string) bool {
	if len(value) != len("sha256:")+64 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	for _, r := range value[len("sha256:"):] {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			return false
		}
	}
	return true
}

// BatchClient is the asynchronous, bulk counterpart to Client. Implementations are
// registered per protocol (RegisterBatchProvider) and constructed via NewBatchClient.
type BatchClient interface {
	// Submit uploads the items and creates the batch, returning a handle to poll.
	// It validates items up front (per-item invariants, duplicate custom_ids,
	// provider-specific homogeneity) and fails before any network effect on a bad set.
	Submit(ctx context.Context, items []BatchItem, opts BatchOptions) (*BatchHandle, error)

	// Recover finds the unique provider batch carrying recoveryKey. A missing key
	// returns ErrBatchNotFound; multiple matches return ErrBatchRecoveryConflict.
	Recover(ctx context.Context, recoveryKey string) (*BatchHandle, error)

	// Get retrieves a refreshed handle from the complete durable handle.
	Get(ctx context.Context, handle BatchHandle) (*BatchHandle, error)

	// Results downloads and parses all resolved lines once the batch has output. Each
	// BatchResult is correlated by ItemID; failed lines carry Error. Calling it
	// before results exist returns whatever is available (possibly empty).
	Results(ctx context.Context, handle BatchHandle) ([]BatchResult, error)

	// Cancel requests cancellation and returns the updated handle.
	Cancel(ctx context.Context, handle BatchHandle) (*BatchHandle, error)

	// Close releases resources (idle HTTP connections).
	Close() error
}

// WaitForBatch polls Get until the batch reaches a terminal status or ctx is
// cancelled. It is caller-driven — pollInterval and ctx are the only controls, never
// a hidden long block — mirroring how Stream hands iteration control to the caller.
// On ctx cancellation it returns the last observed handle together with ctx.Err().
func WaitForBatch(ctx context.Context, bc BatchClient, handle BatchHandle, pollInterval time.Duration) (*BatchHandle, error) {
	if pollInterval <= 0 {
		pollInterval = DefaultBatchPollInterval
	}
	var last *BatchHandle
	for {
		h, err := bc.Get(ctx, handle)
		if err != nil {
			return last, err
		}
		last = h
		handle = *h
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
