package llm

import (
	"context"
	"log/slog"
	"math"
	"strings"
	"sync"
	"time"
)

// ModalityOperation names the modality call a UsageEvent describes.
type ModalityOperation string

// Modality operations reported to Config.UsageHook.
const (
	OperationTranscription   ModalityOperation = "transcription"
	OperationEmbeddings      ModalityOperation = "embeddings"
	OperationImageGeneration ModalityOperation = "image_generation"
	OperationSpeechSynthesis ModalityOperation = "speech_synthesis"
)

// UsageEvent describes one provider attempt of a modality operation. It is
// delivered to Config.UsageHook once per attempt, failed attempts included.
type UsageEvent struct {
	Operation ModalityOperation
	Provider  string
	// Model is the model the attempt targeted (the request's Model when set,
	// otherwise the client's), alias-resolved.
	Model string
	// InputTokens is the provider-reported prompt size, audio included;
	// AudioInputTokens is the audio share of it. OutputTokens includes
	// thinking tokens. All are 0 when the provider reported no usage — they
	// are never estimated.
	InputTokens      int
	OutputTokens     int
	AudioInputTokens int
	// AudioSeconds is the billed audio duration when the provider reports one
	// (duration-billed speech-to-text such as whisper-1 or xAI /v1/stt).
	AudioSeconds float64
	// CostUSD is estimated from registry list prices (EstimateCost). It is 0
	// when a needed price is unknown or the provider reported no usage; a
	// failed attempt that the provider billed anyway keeps its cost.
	CostUSD float64
	Latency time.Duration
	// Err is the attempt's error, nil on success.
	Err error
	// Attempt is the 0-based position in a NewFallbackModalityClient chain
	// (always 0 outside one); Fallback is true when Attempt > 0.
	Attempt  int
	Fallback bool
}

// UsageHook receives one UsageEvent per modality provider attempt. It runs
// synchronously on the calling goroutine after the attempt finishes, so it
// should be fast (hand off to a channel or queue for slow sinks). A panic in
// the hook is recovered and logged; it never changes the call's result.
type UsageHook func(UsageEvent)

// ModalityUsage is what a modality adapter reports about the provider calls it
// made for one operation. Counts are provider-reported, never estimated.
type ModalityUsage struct {
	InputTokens      int
	OutputTokens     int
	AudioInputTokens int
	AudioSeconds     float64
}

type usageRecorderKey struct{}
type modalityAttemptKey struct{}

type usageRecorder struct {
	mu    sync.Mutex
	usage ModalityUsage
}

// ReportModalityUsage records provider-reported usage for the modality
// operation running under ctx. Adapters call it once per provider response;
// repeated reports (one operation spanning several HTTP calls) are summed.
// Negative counts and non-finite or negative durations are dropped, and the
// audio share is capped at the input total, so a malformed provider payload
// yields zero usage rather than garbage. It is a no-op when ctx carries no
// recorder, i.e. outside a client built by NewModalityClient.
func ReportModalityUsage(ctx context.Context, usage ModalityUsage) {
	if ctx == nil {
		return
	}
	recorder, ok := ctx.Value(usageRecorderKey{}).(*usageRecorder)
	if !ok || recorder == nil {
		return
	}
	usage = sanitizeModalityUsage(usage)
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	recorder.usage.InputTokens = saturatingAdd(recorder.usage.InputTokens, usage.InputTokens)
	recorder.usage.OutputTokens = saturatingAdd(recorder.usage.OutputTokens, usage.OutputTokens)
	recorder.usage.AudioInputTokens = saturatingAdd(recorder.usage.AudioInputTokens, usage.AudioInputTokens)
	recorder.usage.AudioSeconds += usage.AudioSeconds
}

func sanitizeModalityUsage(usage ModalityUsage) ModalityUsage {
	if usage.InputTokens < 0 {
		usage.InputTokens = 0
	}
	if usage.OutputTokens < 0 {
		usage.OutputTokens = 0
	}
	if usage.AudioInputTokens < 0 {
		usage.AudioInputTokens = 0
	}
	if usage.AudioInputTokens > usage.InputTokens {
		usage.AudioInputTokens = usage.InputTokens
	}
	if math.IsNaN(usage.AudioSeconds) || math.IsInf(usage.AudioSeconds, 0) || usage.AudioSeconds < 0 {
		usage.AudioSeconds = 0
	}
	return usage
}

func saturatingAdd(a, b int) int {
	if b > 0 && a > math.MaxInt-b {
		return math.MaxInt
	}
	return a + b
}

func (recorder *usageRecorder) snapshot() ModalityUsage {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	return recorder.usage
}

// withModalityAttempt marks ctx with the fallback-chain position of the
// deployment about to be called.
func withModalityAttempt(ctx context.Context, attempt int) context.Context {
	return context.WithValue(ctx, modalityAttemptKey{}, attempt)
}

func modalityAttempt(ctx context.Context) int {
	if attempt, ok := ctx.Value(modalityAttemptKey{}).(int); ok && attempt > 0 {
		return attempt
	}
	return 0
}

// observeModality runs one provider attempt and, when cfg.UsageHook is set,
// reports it. The hook never alters the returned values.
func observeModality[T any](
	ctx context.Context, cfg Config, operation ModalityOperation, requestedModel string,
	call func(context.Context) (T, error),
) (T, error) {
	if cfg.UsageHook == nil {
		return call(ctx)
	}
	recorder := &usageRecorder{}
	attemptCtx := context.WithValue(ctx, usageRecorderKey{}, recorder)
	started := time.Now()
	result, err := call(attemptCtx)
	latency := time.Since(started)

	model := strings.TrimSpace(requestedModel)
	if model == "" {
		model = cfg.Model
	}
	model = ResolveModelAlias(model)
	usage := recorder.snapshot()
	attempt := modalityAttempt(ctx)
	event := UsageEvent{
		Operation:        operation,
		Provider:         cfg.Provider,
		Model:            model,
		InputTokens:      usage.InputTokens,
		OutputTokens:     usage.OutputTokens,
		AudioInputTokens: usage.AudioInputTokens,
		AudioSeconds:     usage.AudioSeconds,
		CostUSD: EstimateCost(CostInput{
			Model:            model,
			InputTokens:      usage.InputTokens,
			OutputTokens:     usage.OutputTokens,
			AudioInputTokens: usage.AudioInputTokens,
			AudioSeconds:     usage.AudioSeconds,
		}),
		Latency:  latency,
		Err:      err,
		Attempt:  attempt,
		Fallback: attempt > 0,
	}
	deliverUsageEvent(cfg.UsageHook, event)
	return result, err
}

func deliverUsageEvent(hook UsageHook, event UsageEvent) {
	defer func() {
		if recovered := recover(); recovered != nil {
			slog.Warn("llm: usage hook panicked; event dropped",
				"component", "llm", "operation", string(event.Operation),
				"provider", event.Provider, "model", event.Model)
		}
	}()
	hook(event)
}
