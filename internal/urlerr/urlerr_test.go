package urlerr_test

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/bds421/rho-llm/internal/urlerr"
)

// net/http prints the request URL in every transport error and strips only
// the password: a key carried as the username or in ?key= reached the error
// text (B32).
func TestRedactScrubsTheRequestURL(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close() // nothing listens: the dial fails
	for _, raw := range []string{
		"http://sk-user-secret@" + addr + "/v1/chat",
		"http://u:pw-secret@" + addr + "/v1/chat",
		"http://" + addr + "/v1/chat?key=q-secret&alt=sse",
	} {
		req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, raw, nil)
		_, doErr := (&http.Client{Timeout: time.Second}).Do(req)
		if doErr == nil {
			t.Fatal("dial unexpectedly succeeded")
		}
		red := urlerr.Redact(doErr)
		wrapped := urlerr.Redact(fmt.Errorf("request failed: %w", doErr))
		for _, e := range []error{red, wrapped} {
			if strings.Contains(e.Error(), "secret") {
				t.Errorf("secret in %q", e)
			}
			var ue *url.Error
			var ne net.Error
			if !errors.As(e, &ue) || !errors.As(e, &ne) {
				t.Errorf("redaction lost the error types: %T", e)
			}
		}
		if !strings.Contains(red.Error(), "/v1/chat") {
			t.Errorf("path lost: %q", red)
		}
	}
	plain := errors.New("x")
	if urlerr.Redact(plain) != plain || urlerr.Redact(nil) != nil {
		t.Error("non-URL errors must pass through")
	}
}
