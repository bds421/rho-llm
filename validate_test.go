package llm_test

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

func weatherTool() llm.Tool {
	return llm.Tool{
		Name: "weather",
		InputSchema: map[string]any{
			"type":     "object",
			"required": []any{"city"},
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
				"days": map[string]any{"type": "integer"},
				"unit": map[string]any{"type": "string", "enum": []any{"c", "f"}},
			},
		},
	}
}

func TestValidateToolCall(t *testing.T) {
	tool := weatherTool()
	cases := []struct {
		name    string
		input   any
		wantErr bool
	}{
		{"valid object", map[string]any{"city": "Paris", "days": 3.0, "unit": "c"}, false},
		{"missing required", map[string]any{"days": 3.0}, true},
		{"wrong type (number for string)", map[string]any{"city": 42.0}, true},
		{"integer non-integral", map[string]any{"city": "x", "days": 3.5}, true},
		{"integer integral float ok", map[string]any{"city": "x", "days": 2.0}, false},
		{"enum violation", map[string]any{"city": "x", "unit": "kelvin"}, true},
		{"enum match", map[string]any{"city": "x", "unit": "f"}, false},
		{"unknown extra property allowed", map[string]any{"city": "x", "extra": true}, false},
		{"raw JSON string arguments", `{"city":"Berlin"}`, false},
		{"raw JSON string invalid", `{not json}`, true},
		{"input not an object", 5.0, true},
		{"nil input but required missing", nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := llm.ValidateToolCall(tool, llm.ToolCall{Name: "weather", Input: tc.input})
			if tc.wantErr && err == nil {
				t.Errorf("ValidateToolCall(%v) = nil, want error", tc.input)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("ValidateToolCall(%v) = %v, want nil", tc.input, err)
			}
		})
	}
}

func TestValidateToolCallNoSchemaIsNoop(t *testing.T) {
	tool := llm.Tool{Name: "freeform"} // no InputSchema
	if err := llm.ValidateToolCall(tool, llm.ToolCall{Input: "anything at all"}); err != nil {
		t.Errorf("no-schema tool should not validate: %v", err)
	}
}
