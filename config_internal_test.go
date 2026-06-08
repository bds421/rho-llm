package llm

import (
	"net/url"
	"testing"
)

// TestSameOriginStripsOnSchemeDowngrade verifies F4: an https→http same-host
// redirect is NOT same-origin (so auth headers are stripped — the old host-only
// check would have kept them and leaked the key over plaintext).
func TestSameOriginStripsOnSchemeDowngrade(t *testing.T) {
	mustURL := func(s string) *url.URL {
		u, err := url.Parse(s)
		if err != nil {
			t.Fatalf("parse %q: %v", s, err)
		}
		return u
	}
	cases := []struct {
		name string
		a, b string
		want bool
	}{
		{"https->http same host (downgrade)", "https://api.x.com/a", "http://api.x.com/a", false},
		{"http->https same host (upgrade)", "http://api.x.com/a", "https://api.x.com/a", false},
		{"same scheme+host, different path", "https://api.x.com/a", "https://api.x.com/b", true},
		{"same scheme, different host", "https://api.x.com/a", "https://evil.x.com/a", false},
		{"different port same host name", "https://api.x.com:443/a", "https://api.x.com:8443/a", false},
		{"identical", "https://api.x.com/a", "https://api.x.com/a", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sameOrigin(mustURL(tc.a), mustURL(tc.b)); got != tc.want {
				t.Errorf("sameOrigin(%q, %q) = %v, want %v (auth header leak risk)", tc.a, tc.b, got, tc.want)
			}
		})
	}
}
