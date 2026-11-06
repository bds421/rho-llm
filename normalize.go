package llm

import (
	"fmt"
	"regexp"
	"strings"
)

// NormalizeForProvider returns a copy of msgs prepared for replay to the given
// provider, applying the cross-provider handoff fixes that every wire protocol
// needs. The input slice and its messages are never mutated.
//
// Transforms applied (in one forward pass + an orphan-repair pass):
//
//   - Errored turns dropped: assistant messages whose StopReason is "error" or
//     "aborted" are removed (an incomplete turn breaks replay).
//   - Thinking degraded across providers: a thinking block is kept verbatim only
//     when replaying to the SAME provider that produced it AND it carries a
//     provider-valid signature (or is redacted). Otherwise its reasoning is
//     preserved by converting it to a plain text block.
//   - Tool-use IDs deduplicated: a duplicate tool_use ID (e.g. Gemini's per-turn
//     synthetic IDs colliding across turns) is rewritten to a unique ID, and the
//     tool_result that answers it is remapped to match — so a handoff to a
//     provider that requires unique IDs (Anthropic) stays valid.
//   - Tool results ordered & pruned: a tool_result is kept only if its tool_use
//     already appeared earlier in the transcript (results before their call, or
//     with no call at all, are dropped).
//   - Orphaned tool calls repaired: a tool_use with no following tool_result gets
//     a synthetic error tool_result, because every provider rejects an
//     unanswered tool call in history.
//   - Server tools stay with their provider: server tool calls, their results
//     and opaque provider blocks are dropped, and citations stripped from text,
//     when the message goes to a provider other than the one that produced it.
//   - Empty messages dropped: a message left with no content is removed.
//
// Sessions apply NormalizeForModel automatically each turn; direct Client users
// can call either before replaying a transcript to a different provider.
func NormalizeForProvider(msgs []Message, provider string) []Message {
	return normalize(msgs, provider, "")
}

// NormalizeForModel is NormalizeForProvider plus one more rule: thinking from a
// DIFFERENT model is dropped instead of replayed or converted to text. A model
// shown another model's reasoning tends to imitate it rather than reason for
// itself. Thinking is kept (subject to the provider rules above) when the
// message has no recorded Model or model is empty.
//
// Two IDs name the same model when their aliases resolve to the same ID or one
// is the other plus a date or version suffix of at least three digits
// ("claude-haiku-4-5" vs "claude-haiku-4-5-20251001", "gemini-x" vs
// "gemini-x-001"; but not "gpt-5" vs "gpt-5-mini" or "claude-opus-4" vs
// "claude-opus-4-8"), because the served model ID a response reports
// can be more specific than the configured one.
func NormalizeForModel(msgs []Message, provider, model string) []Message {
	return normalize(msgs, provider, model)
}

func normalize(msgs []Message, provider, model string) []Message {
	if len(msgs) == 0 {
		return msgs
	}

	usedIDs := make(map[string]bool) // tool_use IDs already emitted (for dedup)
	remap := make(map[string]string) // original tool_use ID -> current emitted ID
	out := make([]Message, 0, len(msgs))

	for _, m := range msgs {
		if m.Role == RoleAssistant && isIncompleteTurn(m.StopReason) {
			continue
		}

		nm := m
		nm.Content = make([]ContentPart, 0, len(m.Content))
		for _, p := range m.Content {
			switch p.Type {
			case ContentToolUse:
				if p.ToolUseID == "" {
					nm.Content = append(nm.Content, p)
					break
				}
				newID := p.ToolUseID
				if usedIDs[newID] {
					newID = uniqueToolID(p.ToolUseID, usedIDs)
				}
				usedIDs[newID] = true
				remap[p.ToolUseID] = newID // results with this original ID now pair to newID
				np := p
				np.ToolUseID = newID
				if m.Provider != provider {
					// A Gemini thought signature is provider-bound, like a thinking
					// signature — it must not cross to a foreign provider.
					np.ThoughtSignature = ""
				}
				nm.Content = append(nm.Content, np)
			case ContentToolResult:
				target, ok := remap[p.ToolResultID]
				if !ok {
					continue // result before its call, or no matching call → drop
				}
				np := p
				np.ToolResultID = target
				nm.Content = append(nm.Content, np)
			case ContentThinking:
				if model != "" && m.Model != "" && !sameModel(m.Model, model) {
					continue // another model's reasoning — drop, don't convert
				}
				sameProvider := m.Provider != "" && m.Provider == provider
				// Replay verbatim only when it would be valid: a redacted block, or
				// a signed block — also with empty text, which is what Claude 4.7+
				// return by default (display "omitted"); dropping it breaks the
				// continuation. Elsewhere a signed-but-empty block has nothing to
				// preserve and is dropped.
				if sameProvider && (p.Redacted || p.ThinkingSignature != "") {
					nm.Content = append(nm.Content, p)
				} else if p.Thinking != "" && !p.Redacted {
					nm.Content = append(nm.Content, ContentPart{Type: ContentText, Text: p.Thinking})
				}
				// else: empty (incl. signed-but-no-text) or redacted-without-replay → drop
			case ContentServerToolUse, ContentServerToolResult, ContentProviderBlock:
				if foreignProvider(m.Provider, provider) {
					continue // provider-run calls and opaque blocks mean nothing elsewhere
				}
				nm.Content = append(nm.Content, p)
			case ContentText:
				if len(p.Citations) > 0 && foreignProvider(m.Provider, provider) {
					p.Citations = nil // citation locations point into provider-held results
				}
				nm.Content = append(nm.Content, p)
			default:
				nm.Content = append(nm.Content, p)
			}
		}

		if len(nm.Content) == 0 {
			continue // nothing survived → drop the now-empty message
		}
		out = append(out, nm)
	}

	// Repair orphaned tool calls. Compute answered IDs across the rebuilt
	// transcript, then insert a synthetic error result after any assistant turn
	// whose (possibly rewritten) tool call is still unanswered.
	answered := make(map[string]bool)
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == ContentToolResult && p.ToolResultID != "" {
				answered[p.ToolResultID] = true
			}
		}
	}

	repaired := make([]Message, 0, len(out))
	for _, m := range out {
		repaired = append(repaired, m)
		if m.Role != RoleAssistant {
			continue
		}
		for _, p := range m.Content {
			if p.Type == ContentToolUse && p.ToolUseID != "" && !answered[p.ToolUseID] {
				repaired = append(repaired, NewToolResultMessage(p.ToolUseID, "No result provided.", true))
				answered[p.ToolUseID] = true // avoid duplicate synthetic results
			}
		}
	}
	return repaired
}

// sameModel reports whether two model IDs name the same model — see
// NormalizeForModel.
func sameModel(a, b string) bool {
	a, b = ResolveModelAlias(a), ResolveModelAlias(b)
	if a == b {
		return true
	}
	return isVersionSuffixOf(a, b) || isVersionSuffixOf(b, a)
}

// versionSuffix matches a dated or numbered snapshot suffix: "-20251001",
// "-001", "-2025-08-07".
var versionSuffix = regexp.MustCompile(`^-[0-9]{3,}(-[0-9]+)*$`)

// isVersionSuffixOf reports whether long is short plus a snapshot suffix.
func isVersionSuffixOf(long, short string) bool {
	rest, ok := strings.CutPrefix(long, short)
	return ok && versionSuffix.MatchString(rest)
}

// uniqueToolID returns the first "<base>#<n>" not already present in used.
func uniqueToolID(base string, used map[string]bool) string {
	for i := 1; ; i++ {
		cand := fmt.Sprintf("%s#%d", base, i)
		if !used[cand] {
			return cand
		}
	}
}

// isIncompleteTurn reports whether a stop reason marks a turn that must not be
// replayed (it never produced a usable assistant message).
func isIncompleteTurn(stopReason string) bool {
	return stopReason == StopError || stopReason == StopAborted
}

// foreignProvider reports whether a message produced by from goes to a
// different provider. A message with no recorded provider is not foreign.
func foreignProvider(from, to string) bool {
	return from != "" && from != to
}
