package llm_test

// Hardening pass 12 — AuthPool "apikey|baseurl" split edge keys.

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestAuthPoolKeyBaseURLSplit(t *testing.T) {
	cases := []struct{ key, wantAPI, wantBase string }{
		{"key", "key", ""},
		{"key|https://base/v1", "key", "https://base/v1"},
		{"key|", "key", ""},
		{"|base", "", "base"}, // empty api (adapter New rejects it downstream)
		{"a|b|c", "a", "b|c"}, // split on FIRST '|' only
	}
	for _, c := range cases {
		pool := llm.NewAuthPool("p", []string{c.key})
		prof, ok := pool.GetCurrent()
		if !ok {
			t.Fatalf("%q: GetCurrent returned false", c.key)
		}
		if prof.APIKey != c.wantAPI || prof.BaseURL != c.wantBase {
			t.Errorf("%q -> APIKey=%q BaseURL=%q, want %q / %q", c.key, prof.APIKey, prof.BaseURL, c.wantAPI, c.wantBase)
		}
	}
}
