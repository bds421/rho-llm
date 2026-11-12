package geminibatch

import "testing"

// A stored batch name with a "/" was sent as the URL path as is, so
// "files/x" or "../models" reached another endpoint with the key (B29).
func TestBatchNameIsCheckedBeforeItBecomesAPath(t *testing.T) {
	for _, tc := range []struct{ stored, id, want string }{
		{"batches/abc-123", "", "batches/abc-123"},
		{"", "abc_123", "batches/abc_123"},
		{"", "batches/abc", "batches/abc"},
	} {
		got, err := batchName(tc.stored, tc.id)
		if err != nil || got != tc.want {
			t.Errorf("batchName(%q, %q) = %q, %v; want %q", tc.stored, tc.id, got, err, tc.want)
		}
	}
	for _, stored := range []string{"files/abc", "batches/../models/gemini", "batches/a/b", "../batches/a", "batches/a?x=1", "batches/", "batches/a:cancel", "/batches/a"} {
		if got, err := batchName(stored, ""); err == nil {
			t.Errorf("hostile name %q accepted as %q", stored, got)
		}
	}
}
