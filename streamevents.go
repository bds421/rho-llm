package llm

import "iter"

// Block-boundary event types emitted by StreamWithBoundaries. They bracket
// contiguous runs of the coarse content/thinking/tool events so UIs can open and
// close blocks cleanly (e.g. render a reasoning panel between ThinkingStart/End).
const (
	EventTextStart     EventType = "text_start"
	EventTextEnd       EventType = "text_end"
	EventThinkingStart EventType = "thinking_start"
	EventThinkingEnd   EventType = "thinking_end"
	EventToolStart     EventType = "tool_start"
	EventToolEnd       EventType = "tool_end"
)

// StreamWithBoundaries wraps a coarse event stream (EventContent / EventThinking /
// EventToolUse / EventDone) and injects block-boundary events around contiguous
// runs: EventTextStart/End around text, EventThinkingStart/End around thinking,
// and EventToolStart/End around each tool call. The original events pass through
// unchanged and errors propagate as-is.
//
// This derives fine-grained boundaries uniformly over the neutral stream, so
// every provider gets identical semantics without each adapter re-implementing
// them. Wrap any Client.Stream result:
//
//	for ev, err := range llm.StreamWithBoundaries(client.Stream(ctx, req)) { ... }
func StreamWithBoundaries(seq iter.Seq2[StreamEvent, error]) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		open := EventType("") // currently open delta block: EventContent, EventThinking, or ""

		closeBlock := func() bool {
			switch open {
			case EventContent:
				open = ""
				return yield(StreamEvent{Type: EventTextEnd}, nil)
			case EventThinking:
				open = ""
				return yield(StreamEvent{Type: EventThinkingEnd}, nil)
			}
			return true
		}

		for ev, err := range seq {
			if err != nil {
				// Close any open block first so every Start has a matching End,
				// even when the underlying stream errors mid-block. If the
				// consumer stops during that injected End event, honor it and do
				// NOT call yield again — calling yield after it has returned false
				// violates the iter.Seq2 contract and can panic composed iterators.
				if !closeBlock() {
					return
				}
				yield(StreamEvent{}, err)
				return
			}
			switch ev.Type {
			case EventContent:
				if open != EventContent {
					if !closeBlock() {
						return
					}
					if !yield(StreamEvent{Type: EventTextStart}, nil) {
						return
					}
					open = EventContent
				}
				if !yield(ev, nil) {
					return
				}
			case EventThinking:
				if open != EventThinking {
					if !closeBlock() {
						return
					}
					if !yield(StreamEvent{Type: EventThinkingStart}, nil) {
						return
					}
					open = EventThinking
				}
				if !yield(ev, nil) {
					return
				}
			case EventToolUse:
				if !closeBlock() {
					return
				}
				if !yield(StreamEvent{Type: EventToolStart}, nil) {
					return
				}
				if !yield(ev, nil) {
					return
				}
				if !yield(StreamEvent{Type: EventToolEnd}, nil) {
					return
				}
			case EventDone:
				if !closeBlock() {
					return
				}
				if !yield(ev, nil) {
					return
				}
			default:
				if !yield(ev, nil) {
					return
				}
			}
		}
		// Stream ended without an explicit Done — close any still-open block.
		closeBlock()
	}
}
