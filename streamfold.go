package llm

import (
	"fmt"
	"io"
	"iter"
	"strings"
)

// StreamAccumulator folds stream events into a Response: text, thinking,
// tool calls, and the usage, stop reason and assembled content of the
// terminal event. Session.Stream folds with it; CollectStream wraps it for
// callers that want a Response from a stream.
type StreamAccumulator struct {
	text, thinking   strings.Builder
	thinkingSig      string
	thinkingRedacted bool
	toolCalls        []ToolCall
	final            StreamEvent
	done             bool
}

// Add folds one event. An EventError yielded together with an error carries
// the usage seen before the stream broke (Anthropic); add it like any other.
func (a *StreamAccumulator) Add(ev StreamEvent) {
	switch ev.Type {
	case EventContent:
		a.text.WriteString(ev.Text)
	case EventThinking:
		a.thinking.WriteString(ev.Thinking)
	case EventToolUse:
		if ev.ToolCall != nil {
			a.toolCalls = append(a.toolCalls, *ev.ToolCall)
		}
	case EventDone:
		a.final, a.done = ev, true
		if ev.ThinkingSignature != "" {
			a.thinkingSig = ev.ThinkingSignature
		}
		if ev.ThinkingRedacted {
			a.thinkingRedacted = true
		}
	case EventError:
		a.final = ev
	}
}

// Done reports whether the terminal EventDone arrived.
func (a *StreamAccumulator) Done() bool { return a.done }

// Response returns the folded turn. Before EventDone it is partial: the
// content so far and whatever usage the provider reported.
func (a *StreamAccumulator) Response() *Response {
	f := a.final
	r := &Response{
		Model:                 f.Model,
		Content:               a.text.String(),
		Thinking:              a.thinking.String(),
		ThinkingSignature:     a.thinkingSig,
		ThinkingRedacted:      a.thinkingRedacted,
		ToolCalls:             append([]ToolCall(nil), a.toolCalls...),
		StopReason:            f.StopReason,
		RawStopReason:         f.RawStopReason,
		InputTokens:           f.InputTokens,
		OutputTokens:          f.OutputTokens,
		ThinkingTokens:        f.ThinkingTokens,
		CacheCreationTokens:   f.CacheCreationTokens,
		CacheReadTokens:       f.CacheReadTokens,
		CacheCreation1hTokens: f.CacheCreation1hTokens,
		StopDetails:           f.StopDetails,
		ServerToolUse:         f.ServerToolUse,
		Iterations:            f.Iterations,
		ContainerID:           f.ContainerID,
		ReasoningField:        f.ReasoningField,
		InputTransformations:  f.InputTransformations,
	}
	// An adapter that assembles the turn (Anthropic) sends it as Parts; the
	// flat fields then come from those, not from the deltas.
	if len(f.Parts) > 0 {
		r.SetParts(f.Parts)
	}
	return r
}

// CollectStream drains a stream into a Response. On error it returns the
// partial Response (content and the usage seen so far) together with the
// error, so the caller can record what a broken turn cost.
func CollectStream(seq iter.Seq2[StreamEvent, error]) (*Response, error) {
	var acc StreamAccumulator
	for ev, err := range seq {
		acc.Add(ev)
		if err != nil {
			return acc.Response(), err
		}
	}
	if !acc.Done() {
		return acc.Response(), ErrStreamIncomplete
	}
	return acc.Response(), nil
}

// ErrStreamIncomplete is returned by CollectStream when the stream ended
// without its terminal event.
var ErrStreamIncomplete = fmt.Errorf("llm: stream ended without completion: %w", io.ErrUnexpectedEOF)
