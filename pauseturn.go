package llm

import (
	"errors"
	"fmt"
)

// ErrNotPaused is returned when a pause_turn continuation is requested for a
// turn that did not stop with StopPauseTurn.
var ErrNotPaused = errors.New("llm: the last turn is not paused (stop reason is not pause_turn)")

// ContinuePausedTurn returns the request that continues a paused turn: req
// with resp's assistant content appended unchanged, as the provider
// documents (no extra user message). Every other field of req — Tools,
// ServerTools, System, cache flags — is left as it is, so the request prefix
// encodes to the same bytes and the prompt cache stays valid; pass the same
// req that produced resp (or the previous continuation). The container is
// carried over from resp when req names none. A continuation can pause again:
// call ContinuePausedTurn on each paused response, capping the number of
// rounds as for any retry loop.
//
// It returns ErrNotPaused when resp did not stop with StopPauseTurn. req's
// Messages slice is copied, never appended to in place.
func ContinuePausedTurn(req Request, resp *Response) (Request, error) {
	if resp == nil {
		return req, fmt.Errorf("llm: ContinuePausedTurn: nil response")
	}
	if resp.StopReason != StopPauseTurn {
		return req, ErrNotPaused
	}
	msgs := make([]Message, 0, len(req.Messages)+1)
	msgs = append(msgs, req.Messages...)
	msgs = append(msgs, NewAssistantMessage(resp))
	req.Messages = msgs
	if req.ContainerID == "" {
		req.ContainerID = resp.ContainerID
	}
	return req, nil
}

// pausedTurn reports whether msgs ends with a paused assistant turn, and the
// container that turn ran in.
func pausedTurn(msgs []Message) (bool, string) {
	if len(msgs) == 0 {
		return false, ""
	}
	last := msgs[len(msgs)-1]
	if last.Role != RoleAssistant || last.StopReason != StopPauseTurn {
		return false, ""
	}
	return true, last.ContainerID
}
