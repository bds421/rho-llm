package llm

import (
	"encoding/json"
	"fmt"
	"slices"
	"sort"
)

// MergeSamplingParams marshals an adapter's request struct and merges
// Request.SamplingParams into the resulting JSON object at the top level.
//
// This is the single implementation behind the provider-specific request-body
// passthrough (see Request.SamplingParams); every adapter calls it at the point
// it would otherwise call json.Marshal on its request struct, so the merge rules
// — and the protections below — are identical across all four wire protocols.
//
// Protections, in order:
//
//   - A key already present in the marshaled body is REJECTED. Those are the
//     fields the adapter itself set, so a passthrough key cannot redirect the
//     request to a different model, rewrite its messages, or swap its tools.
//     This is deliberately stricter than "last write wins": silently overriding
//     "model" or "messages" is exactly the failure a caller would never see.
//   - A key that is reserved but absent from this particular body (because it was
//     omitempty and unset) is rejected too, so the guarantee does not depend on
//     whether an optional field happened to be populated on this request.
//   - An empty key is rejected: it is never a valid wire field and would produce
//     a body no provider can parse.
//   - A value that cannot be marshaled is rejected, with the offending key named.
//
// A nil or empty SamplingParams map returns the plain marshaling, so the
// non-passthrough path costs one map check.
func MergeSamplingParams(apiReq any, params map[string]any) ([]byte, error) {
	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, err
	}
	if len(params) == 0 {
		return body, nil
	}

	var merged map[string]json.RawMessage
	if err := json.Unmarshal(body, &merged); err != nil {
		// The adapter's request struct must marshal to a JSON object for a
		// top-level merge to mean anything.
		return nil, fmt.Errorf("llm: cannot merge SamplingParams into a non-object request body: %w", err)
	}
	if merged == nil {
		return nil, fmt.Errorf("llm: cannot merge SamplingParams into a non-object request body: null")
	}

	// Iterate in sorted order so a request with several offending keys always
	// reports the same one — an error that changes between identical runs is
	// far harder to debug than a deterministic one.
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		if k == "" {
			return nil, fmt.Errorf("llm: SamplingParams contains an empty key")
		}
		if _, taken := merged[k]; taken {
			return nil, fmt.Errorf("llm: SamplingParams key %q is already set by the adapter; use the typed Request field instead", k)
		}
		if slices.Contains(reservedBodyKeys, k) {
			return nil, fmt.Errorf("llm: SamplingParams key %q is reserved by the request builder; use the typed Request field instead", k)
		}
		raw, err := json.Marshal(params[k])
		if err != nil {
			return nil, fmt.Errorf("llm: SamplingParams key %q: %w", k, err)
		}
		merged[k] = raw
	}

	return json.Marshal(merged)
}

// reservedBodyKeys are wire fields that define the shape and target of a request
// across the four protocols. They are refused even when absent from a given body
// (an omitempty field that happened to be unset), so the guarantee holds for
// every request rather than only the ones where the field was populated.
//
// Only structural fields belong here — the ones whose override would change what
// is asked, of which model, with what history or tools. Tuning knobs
// ("temperature", "top_p", "seed", "priority", "provider", …) are exactly what
// the passthrough exists to carry and are deliberately NOT reserved.
var reservedBodyKeys = []string{
	// OpenAI Chat Completions / openai_compat
	"messages", "model", "stream", "stream_options", "tools", "tool_choice",
	// OpenAI Responses
	"input", "instructions",
	// Anthropic Messages
	"system", "thinking",
	// Gemini generateContent
	"contents", "systemInstruction", "generationConfig", "toolConfig",
}
