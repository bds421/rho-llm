package llm_test

import (
	"math"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// ValidateToolCall must never PANIC, even on a hostile schema. An `enum`
// containing an uncomparable value (a slice) would panic with a naive `==`.
func TestValidateToolCallEnumUncomparableNoPanic(t *testing.T) {
	tool := llm.Tool{
		Name: "t",
		InputSchema: map[string]any{
			"properties": map[string]any{
				// enum entries are slices (uncomparable with ==)
				"p": map[string]any{"enum": []any{[]any{"x"}, []any{"y"}}},
			},
		},
	}
	// arg value is also a slice — same dynamic type as the enum entries.
	call := llm.ToolCall{Name: "t", Input: map[string]any{"p": []any{"x"}}}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ValidateToolCall panicked on uncomparable enum: %v", r)
		}
	}()
	_ = llm.ValidateToolCall(tool, call) // must return (nil or error), never panic
}

// ValidateToolCall with a `required` list holding non-strings must not panic.
func TestValidateToolCallHostileRequiredNoPanic(t *testing.T) {
	tool := llm.Tool{
		Name: "t",
		InputSchema: map[string]any{
			"required": []any{123, []any{"nested"}, nil, "realfield"},
			"properties": map[string]any{
				"realfield": map[string]any{"type": "string"},
			},
		},
	}
	call := llm.ToolCall{Name: "t", Input: map[string]any{"realfield": "v"}}
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ValidateToolCall panicked on hostile required: %v", r)
		}
	}()
	_ = llm.ValidateToolCall(tool, call)
}

// ValidateToolCall must reject non-finite floats for an integer field — +Inf
// would otherwise pass (math.Trunc(Inf)==Inf) and serialize to invalid JSON.
func TestValidateToolCallRejectsNonFiniteInteger(t *testing.T) {
	tool := llm.Tool{Name: "t", InputSchema: map[string]any{
		"properties": map[string]any{"n": map[string]any{"type": "integer"}},
	}}
	if err := llm.ValidateToolCall(tool, llm.ToolCall{Name: "t", Input: map[string]any{"n": math.Inf(1)}}); err == nil {
		t.Error("ValidateToolCall accepted +Inf as a valid integer")
	}
	if err := llm.ValidateToolCall(tool, llm.ToolCall{Name: "t", Input: map[string]any{"n": math.NaN()}}); err == nil {
		t.Error("ValidateToolCall accepted NaN as a valid integer")
	}
}

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

// A schema written in Go (as the README and examples do) uses []string for
// "required" and "enum", []int for numeric enums and typed property maps. The
// checks asserted the JSON-decoded shapes only, so all of them were skipped
// (B9).
func TestValidateToolCallGoAuthoredSchema(t *testing.T) {
	tool := llm.Tool{Name: "book", InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]map[string]any{
			"city":   {"type": "string", "enum": []string{"Graz", "Wien"}},
			"guests": {"type": "integer", "enum": []int{1, 2}},
		},
		"required": []string{"city"},
	}}
	cases := []struct {
		input   any
		wantErr string
	}{
		{map[string]any{"city": "Graz", "guests": 2.0}, ""},
		{`{"city":"Wien","guests":1}`, ""},
		{map[string]any{"guests": 1.0}, `missing required argument "city"`},
		{map[string]any{"city": "Linz"}, `"city"`},
		{map[string]any{"city": "Graz", "guests": 3.0}, `"guests"`},
		{map[string]any{"city": 7.0}, `should be string`},
	}
	for _, tc := range cases {
		err := llm.ValidateToolCall(tool, llm.ToolCall{Name: "book", Input: tc.input})
		switch {
		case tc.wantErr == "" && err != nil:
			t.Errorf("%v: unexpected error %v", tc.input, err)
		case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
			t.Errorf("%v: err = %v, want one mentioning %s", tc.input, err, tc.wantErr)
		}
	}
	// An unmarshalable schema is reported, not silently accepted.
	bad := llm.Tool{Name: "bad", InputSchema: map[string]any{"type": "object", "x": func() {}}}
	if err := llm.ValidateToolCall(bad, llm.ToolCall{Input: map[string]any{}}); err == nil {
		t.Error("schema that cannot be encoded was accepted")
	}
}
