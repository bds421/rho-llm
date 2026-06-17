package llm

// Hardening pass 2 — OAuth 2.0 device flow (oauth.go).

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// The RFC-mandated error_description must reach the caller, not be discarded —
// "invalid_client" alone is far less actionable than the server's reason.
func TestStartDeviceAuthSurfacesErrorDescription(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":"invalid_client","error_description":"the supplied client id is unknown"}`)
	}))
	defer srv.Close()

	_, err := StartDeviceAuth(context.Background(), DeviceAuthConfig{ClientID: "x", DeviceAuthURL: srv.URL, TokenURL: srv.URL})
	if err == nil {
		t.Fatal("want an error on a 400 device-authorization response")
	}
	if !strings.Contains(err.Error(), "the supplied client id is unknown") {
		t.Fatalf("error must surface error_description, got: %v", err)
	}
}

// A server-supplied interval must be clamped to a sane range, overflow-safe — a
// hostile huge value must NOT wrap past int64 into a tiny or negative poll delay
// that defeats the minimum.
func TestClampPollInterval(t *testing.T) {
	cases := []struct {
		in   int
		want time.Duration
	}{
		{0, defaultPollInterval},
		{-10, defaultPollInterval},
		{1, 1 * time.Second},
		{3, 3 * time.Second},
		{100000, maxPollInterval},
		{math.MaxInt, maxPollInterval}, // must clamp, never overflow to a tiny/negative duration
	}
	for _, c := range cases {
		got := clampPollInterval(c.in)
		if got != c.want {
			t.Errorf("clampPollInterval(%d) = %v, want %v", c.in, got, c.want)
		}
		if got <= 0 {
			t.Errorf("clampPollInterval(%d) = %v — must be a positive delay (overflow/clamp bug)", c.in, got)
		}
	}
}

// PollDeviceToken must stop when the device code expires (expires_in), as its doc
// promises — not poll until the caller's context deadline. A server stuck on
// authorization_pending must not make it hang.
func TestPollDeviceTokenStopsAtExpiry(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":"authorization_pending"}`) // never authorizes
	}))
	defer srv.Close()

	// Generous ctx so the test distinguishes "stopped at expiry" from "stopped at ctx".
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
	defer cancel()
	da := &DeviceAuth{DeviceCode: "dc", Interval: 1, ExpiresIn: 1}

	start := time.Now()
	_, err := PollDeviceToken(ctx, DeviceAuthConfig{ClientID: "x", TokenURL: srv.URL}, da)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("want an error when the grant expires before authorization")
	}
	if errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("PollDeviceToken hit the ctx deadline instead of enforcing expires_in (hang risk): %v", err)
	}
	if elapsed > 4*time.Second {
		t.Fatalf("PollDeviceToken took %v — it did not stop at expires_in", elapsed)
	}
}
