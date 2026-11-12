package toolinput_test

import (
	"reflect"
	"testing"

	"github.com/bds421/rho-llm/internal/toolinput"
)

// A tool without parameters arrives as "" (Anthropic sends no partial_json, or
// an empty one; OpenAI-compatible servers send "arguments":""). It was stored
// as the string "" and replayed as "input":"" / "arguments":"\"\"", which the
// next request carries forever (B8).
func TestDecode(t *testing.T) {
	empty := map[string]any{}
	cases := []struct {
		raw     string
		want    any
		wantErr bool
	}{
		{"", empty, false},
		{"   \n", empty, false},
		{"null", empty, false},
		{" null ", empty, false},
		{"{}", empty, false},
		{`{"city":"Graz"}`, map[string]any{"city": "Graz"}, false},
		{`[1,2]`, []any{1.0, 2.0}, false},
		{`"text"`, "text", false},
		{`{"city":`, `{"city":`, true}, // truncated at max_tokens: kept raw
		{`not json`, `not json`, true},
	}
	for _, tc := range cases {
		got, err := toolinput.Decode(tc.raw)
		if (err != nil) != tc.wantErr || !reflect.DeepEqual(got, tc.want) {
			t.Errorf("Decode(%q) = %#v, %v; want %#v, err=%v", tc.raw, got, err, tc.want, tc.wantErr)
		}
	}
}

func TestObject(t *testing.T) {
	empty := map[string]any{}
	obj := map[string]any{"a": 1.0}
	cases := []struct {
		in   any
		want any
	}{
		{nil, empty},
		{"", empty},
		{"  ", empty},
		{"null", empty},
		{`{"a":1}`, obj},
		{obj, obj},
		{`{"a":`, `{"a":`}, // malformed text is not invented into an object
		{`[1]`, `[1]`},
		{42, 42},
	}
	for _, tc := range cases {
		if got := toolinput.Object(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("Object(%#v) = %#v, want %#v", tc.in, got, tc.want)
		}
	}
}
