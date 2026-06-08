package llm

import (
	"context"
	"iter"
)

// CompleteSimple is a one-shot convenience over Client.Complete: it sends a single
// user prompt and returns the response. Generation settings (model, max tokens,
// thinking level) come from the client's Config.
func CompleteSimple(ctx context.Context, client Client, prompt string) (*Response, error) {
	return client.Complete(ctx, Request{
		Messages: []Message{NewTextMessage(RoleUser, prompt)},
	})
}

// StreamSimple is the streaming counterpart of CompleteSimple.
func StreamSimple(ctx context.Context, client Client, prompt string) iter.Seq2[StreamEvent, error] {
	return client.Stream(ctx, Request{
		Messages: []Message{NewTextMessage(RoleUser, prompt)},
	})
}
