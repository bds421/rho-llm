package llm

import (
	"encoding/json"
	"strings"
)

// DecodeToolInput decodes the argument text of a tool call as an adapter
// receives it (a complete "arguments" string or the concatenated stream
// deltas). A tool without parameters arrives as "" or "null" from several
// providers (an empty Anthropic partial_json, "arguments":"" from
// OpenAI-compatible servers); that is an empty object, not a string. Malformed
// JSON returns the raw text together with the parse error, so the caller can
// log it and keep what the model sent.
func DecodeToolInput(raw string) (any, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" || trimmed == "null" {
		return map[string]any{}, nil
	}
	var input any
	if err := json.Unmarshal([]byte(trimmed), &input); err != nil {
		return raw, err
	}
	if input == nil {
		return map[string]any{}, nil
	}
	return input, nil
}

// ToolInputObject returns the tool input an adapter sends back to a provider
// when it replays a tool call. Every provider expects an object there; a nil
// input or an empty or "null" string (as transcripts persisted before
// DecodeToolInput existed may hold) becomes an empty object, and a string that
// holds a JSON object becomes that object. Any other value is returned as is.
func ToolInputObject(v any) any {
	switch in := v.(type) {
	case nil:
		return map[string]any{}
	case string:
		trimmed := strings.TrimSpace(in)
		if trimmed == "" || trimmed == "null" {
			return map[string]any{}
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(trimmed), &obj); err == nil && obj != nil {
			return obj
		}
	}
	return v
}
