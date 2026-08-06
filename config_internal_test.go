package llm

import (
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestSafeHTTPClientPreservesStandardProxyDiscovery(t *testing.T) {
	transport, ok := SafeHTTPClient(time.Second).Transport.(*http.Transport)
	if !ok {
		t.Fatal("SafeHTTPClient transport is not *http.Transport")
	}
	if transport.Proxy == nil {
		t.Fatal("SafeHTTPClient disables HTTP_PROXY and HTTPS_PROXY discovery")
	}
	if reflect.ValueOf(transport.Proxy).Pointer() != reflect.ValueOf(http.ProxyFromEnvironment).Pointer() {
		t.Fatal("SafeHTTPClient does not use the standard environment proxy policy")
	}
}

func TestNewSafeHTTPClientUsesExplicitProxyInsteadOfAmbientPolicy(t *testing.T) {
	t.Setenv("HTTP_PROXY", "http://ambient.invalid:8080")
	t.Setenv("HTTPS_PROXY", "http://ambient.invalid:8080")

	client, err := NewSafeHTTPClient(Config{
		Timeout:  time.Second,
		ProxyURL: "http://reviewed-proxy.example:3128",
	})
	if err != nil {
		t.Fatal(err)
	}
	transport, ok := client.Transport.(*http.Transport)
	if !ok || transport.Proxy == nil {
		t.Fatal("explicit proxy client does not have a proxy policy")
	}
	req, err := http.NewRequest(http.MethodGet, "https://provider.example/v1/models", nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := transport.Proxy(req)
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.String() != "http://reviewed-proxy.example:3128" {
		t.Fatalf("proxy = %v, want reviewed proxy", got)
	}
}

func TestNewSafeHTTPClientCanExplicitlyBypassAmbientProxy(t *testing.T) {
	t.Setenv("HTTP_PROXY", "http://ambient.invalid:8080")
	t.Setenv("HTTPS_PROXY", "http://ambient.invalid:8080")

	client, err := NewSafeHTTPClient(Config{Timeout: time.Second, DisableProxy: true})
	if err != nil {
		t.Fatal(err)
	}
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatal("proxy-disabled client transport is not *http.Transport")
	}
	if transport.Proxy != nil {
		t.Fatal("proxy-disabled client still consults an ambient proxy policy")
	}
}

func TestNewSafeHTTPClientRejectsAmbiguousOrUnsafeProxyConfiguration(t *testing.T) {
	tests := []Config{
		{ProxyURL: "http://proxy.example:3128", DisableProxy: true},
		{ProxyURL: "socks5://proxy.example:1080"},
		{ProxyURL: "http://user:secret@proxy.example:3128"},
		{ProxyURL: "http://proxy.example:3128/tenant"},
		{ProxyURL: "http://proxy.example:3128?tenant=a"},
		{ProxyURL: "http:///missing-host"},
	}
	for _, cfg := range tests {
		if _, err := NewSafeHTTPClient(cfg); err == nil {
			t.Fatalf("NewSafeHTTPClient(%+v) accepted unsafe proxy configuration", cfg)
		} else if !strings.Contains(err.Error(), "ProxyURL") {
			t.Fatalf("NewSafeHTTPClient(%+v) error = %v", cfg, err)
		}
	}
}

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
