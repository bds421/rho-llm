package llm

// Hardening pass 5 — SSRF guard (CheckBaseURL) host-normalization bypass.

import "testing"

// A trailing dot makes an absolute FQDN that resolvers still map to loopback
// ("localhost." -> 127.0.0.1, "127.0.0.1." -> 127.0.0.1), but it evades both the
// literal "localhost" comparison and net.ParseIP. The SSRF guard must normalize
// the host so these don't slip through.
func TestCheckBaseURLBlocksTrailingDotBypass(t *testing.T) {
	blocked := []string{
		"http://localhost./v1",
		"http://localhost.:8080/v1",
		"http://LOCALHOST./v1",
		"http://127.0.0.1./v1",
		"http://[::1]/v1", // sanity: IPv6 loopback already blocked
	}
	for _, bad := range blocked {
		if err := CheckBaseURL(Config{BaseURL: bad, BlockPrivateBaseURL: true}); err == nil {
			t.Errorf("CheckBaseURL allowed %q — SSRF-guard bypass", bad)
		}
	}
	// A genuine public host must still be allowed.
	if err := CheckBaseURL(Config{BaseURL: "https://api.openai.com/v1", BlockPrivateBaseURL: true}); err != nil {
		t.Errorf("CheckBaseURL wrongly blocked a public host: %v", err)
	}
}
