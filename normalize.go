package llm

import "fmt"

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
//   - Empty messages dropped: a message left with no content is removed.
//
// Sessions apply this automatically each turn; direct Client users can call it
// before replaying a transcript to a different provider.
func NormalizeForProvider(msgs []Message, provider string) []Message {
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
				sameProvider := m.Provider != "" && m.Provider == provider
				if sameProvider && (p.ThinkingSignature != "" || p.Redacted) {
					nm.Content = append(nm.Content, p) // replay verbatim
				} else if p.Thinking != "" && !p.Redacted {
					nm.Content = append(nm.Content, ContentPart{Type: ContentText, Text: p.Thinking})
				}
				// else: empty or redacted-without-replay → drop
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
