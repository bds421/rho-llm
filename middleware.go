package llm

import (
	"context"
	"iter"
	"log/slog"
	"time"
)

// LoggingClient wraps any Client with request/response metadata logging.
// It does NOT log message content (privacy safe) -- only metadata like
// provider, model, message count, tool count, token usage, and errors.
type LoggingClient struct {
	inner  Client
	logger *slog.Logger
}

// WithLogging wraps a Client with request/response logging using the default "llm" component.
func WithLogging(client Client) Client {
	return &LoggingClient{inner: client, logger: slog.Default().With("component", "llm")}
}

// WithLoggingPrefix wraps a Client with request/response logging using a custom component name.
func WithLoggingPrefix(client Client, prefix string) Client {
	return &LoggingClient{inner: client, logger: slog.Default().With("component", prefix)}
}

// Complete logs request metadata, delegates to the inner client, then logs the response.
func (l *LoggingClient) Complete(ctx context.Context, req Request) (*Response, error) {
	model := req.Model
	if model == "" {
		model = l.inner.Model()
	}

	l.logger.Debug("complete request",
		"provider", l.inner.Provider(), "model", model,
		"messages", len(req.Messages), "tools", len(req.Tools), "max_tokens", req.MaxTokens)

	start := time.Now()
	resp, err := l.inner.Complete(ctx, req)
	elapsed := time.Since(start)

	if err != nil {
		l.logger.Warn("complete failed",
			"provider", l.inner.Provider(), "model", model,
			"elapsed", elapsed.Round(time.Millisecond), "error", err)
		return resp, err
	}

	cost := EstimateCost(model, resp.InputTokens, resp.OutputTokens)
	l.logger.Debug("complete done",
		"provider", l.inner.Provider(), "model", model,
		"elapsed", elapsed.Round(time.Millisecond),
		"tokens_in", resp.InputTokens, "tokens_out", resp.OutputTokens,
		"stop", resp.StopReason, "cost", cost)

	return resp, nil
}

// Stream wraps the inner client's stream iterator with metadata logging.
func (l *LoggingClient) Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		model := req.Model
		if model == "" {
			model = l.inner.Model()
		}

		l.logger.Debug("stream request",
			"provider", l.inner.Provider(), "model", model,
			"messages", len(req.Messages), "tools", len(req.Tools), "max_tokens", req.MaxTokens)

		start := time.Now()
		var chunks int
		var lastInputTokens, lastOutputTokens int
		var lastStopReason string
		var streamErr error

		defer func() {
			elapsed := time.Since(start)
			if streamErr != nil {
				l.logger.Warn("stream failed",
					"provider", l.inner.Provider(), "model", model,
					"elapsed", elapsed.Round(time.Millisecond), "chunks", chunks, "error", streamErr)
				return
			}
			cost := EstimateCost(model, lastInputTokens, lastOutputTokens)
			l.logger.Debug("stream done",
				"provider", l.inner.Provider(), "model", model,
				"elapsed", elapsed.Round(time.Millisecond),
				"chunks", chunks, "tokens_in", lastInputTokens, "tokens_out", lastOutputTokens,
				"stop", lastStopReason, "cost", cost)
		}()

		for event, err := range l.inner.Stream(ctx, req) {
			if err != nil {
				streamErr = err
				yield(StreamEvent{}, err)
				return
			}
			switch event.Type {
			case EventContent:
				chunks++
			case EventDone:
				lastStopReason = event.StopReason
				lastInputTokens = event.InputTokens
				lastOutputTokens = event.OutputTokens
			}
			if !yield(event, nil) {
				return
			}
		}
	}
}

// Provider delegates to the inner client.
func (l *LoggingClient) Provider() string {
	return l.inner.Provider()
}

// Model delegates to the inner client.
func (l *LoggingClient) Model() string {
	return l.inner.Model()
}

// Close delegates to the inner client.
func (l *LoggingClient) Close() error {
	return l.inner.Close()
}
