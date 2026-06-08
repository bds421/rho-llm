package llm

import (
	"encoding/json"
	"fmt"
	"math"
)

// ValidateToolCall checks a tool call's arguments against the tool's InputSchema.
// It is best-effort structural validation over a common JSON-Schema subset —
// object `required`, per-property `type`, and `enum` — not a full JSON-Schema
// validator. It returns nil when the tool has no schema (nothing to check).
//
// Use it in a tool loop to reject malformed model arguments before executing:
//
//	for _, call := range resp.ToolCalls {
//	    if err := llm.ValidateToolCall(tool, call); err != nil { ... }
//	}
func ValidateToolCall(tool Tool, call ToolCall) error {
	schema := tool.InputSchema
	if len(schema) == 0 {
		return nil
	}

	args, err := toArgsMap(call.Input)
	if err != nil {
		return fmt.Errorf("tool %q: %w", tool.Name, err)
	}

	// required
	if req, ok := schema["required"].([]any); ok {
		for _, r := range req {
			name, _ := r.(string)
			if name == "" {
				continue
			}
			if _, present := args[name]; !present {
				return fmt.Errorf("tool %q: missing required argument %q", tool.Name, name)
			}
		}
	}

	// per-property type + enum
	props, _ := schema["properties"].(map[string]any)
	for name, val := range args {
		propSchema, ok := props[name].(map[string]any)
		if !ok {
			continue // unknown property — not enforced (no additionalProperties handling)
		}
		if declared, ok := propSchema["type"].(string); ok {
			if !matchesJSONType(declared, val) {
				return fmt.Errorf("tool %q: argument %q should be %s, got %s", tool.Name, name, declared, jsonTypeName(val))
			}
		}
		if enum, ok := propSchema["enum"].([]any); ok && !enumContains(enum, val) {
			return fmt.Errorf("tool %q: argument %q = %v is not one of the allowed enum values", tool.Name, name, val)
		}
	}
	return nil
}

// toArgsMap normalizes a tool call's Input (which adapters may deliver as a
// map[string]any or a raw JSON string) into a map.
func toArgsMap(input any) (map[string]any, error) {
	switch v := input.(type) {
	case nil:
		return map[string]any{}, nil
	case map[string]any:
		return v, nil
	case string:
		var m map[string]any
		if err := json.Unmarshal([]byte(v), &m); err != nil {
			return nil, fmt.Errorf("arguments are not a valid JSON object: %w", err)
		}
		return m, nil
	default:
		return nil, fmt.Errorf("arguments must be a JSON object, got %s", jsonTypeName(input))
	}
}

// matchesJSONType reports whether v matches a declared JSON-Schema type. JSON
// numbers decode to float64, so "integer" accepts an integral float64.
func matchesJSONType(declared string, v any) bool {
	switch declared {
	case "string":
		_, ok := v.(string)
		return ok
	case "boolean":
		_, ok := v.(bool)
		return ok
	case "number":
		return isNumber(v)
	case "integer":
		switch n := v.(type) {
		case float64:
			return n == math.Trunc(n)
		case int, int64:
			return true
		case json.Number:
			f, err := n.Float64()
			return err == nil && f == math.Trunc(f)
		default:
			return false
		}
	case "array":
		_, ok := v.([]any)
		return ok
	case "object":
		_, ok := v.(map[string]any)
		return ok
	case "null":
		return v == nil
	default:
		return true // unknown declared type — don't reject
	}
}

func isNumber(v any) bool {
	switch v.(type) {
	case float64, float32, int, int64, json.Number:
		return true
	default:
		return false
	}
}

func jsonTypeName(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case string:
		return "string"
	case bool:
		return "boolean"
	case float64, float32, int, int64, json.Number:
		return "number"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	default:
		return fmt.Sprintf("%T", v)
	}
}

func enumContains(enum []any, v any) bool {
	for _, e := range enum {
		if e == v {
			return true
		}
	}
	return false
}
