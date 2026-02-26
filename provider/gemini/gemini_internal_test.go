package gemini

import "testing"

// TestMakeToolCallIDResolveToolNameRoundTrip verifies that resolveToolName
// correctly inverts makeToolCallID for various tool name patterns.
func TestMakeToolCallIDResolveToolNameRoundTrip(t *testing.T) {
	tests := []struct {
		index int
		name  string
	}{
		{0, "get_weather"},
		{1, "search"},
		{0, "tool_with_many_underscores"},
		{5, "a"},
		{0, "123_numeric_prefix"},
		{0, "99"},
		{10, "deeply_nested_tool_name"},
	}

	for _, tc := range tests {
		id := makeToolCallID(tc.index, tc.name)
		got := resolveToolName(id)
		if got != tc.name {
			t.Errorf("roundtrip(%d, %q): makeToolCallID=%q, resolveToolName=%q, want %q",
				tc.index, tc.name, id, got, tc.name)
		}
	}
}

// TestResolveToolNameLegacyFormat verifies the legacy "call_<name>" format.
func TestResolveToolNameLegacyFormat(t *testing.T) {
	got := resolveToolName("call_my_tool")
	if got != "my_tool" {
		t.Errorf("resolveToolName(call_my_tool) = %q, want my_tool", got)
	}
}

// TestResolveToolNameNonSynthetic verifies non-call_ IDs pass through.
func TestResolveToolNameNonSynthetic(t *testing.T) {
	got := resolveToolName("toolu_01234abcdef")
	if got != "toolu_01234abcdef" {
		t.Errorf("resolveToolName(toolu_...) = %q, want passthrough", got)
	}
}
