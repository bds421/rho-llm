package llm_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
)

func TestDeviceFlowSuccessWithPending(t *testing.T) {
	// Token endpoint returns authorization_pending on the first poll, then the token.
	var polls int32
	mux := http.NewServeMux()
	mux.HandleFunc("/device", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// interval:1 keeps the test fast.
		w.Write([]byte(`{"device_code":"DEV","user_code":"WXYZ","verification_uri":"https://example.com/device","interval":1}`))
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if atomic.AddInt32(&polls, 1) == 1 {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":"authorization_pending"}`))
			return
		}
		w.Write([]byte(`{"access_token":"ACCESS123","token_type":"bearer","expires_in":3600}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cfg := llm.DeviceAuthConfig{ClientID: "cid", DeviceAuthURL: srv.URL + "/device", TokenURL: srv.URL + "/token", Scope: "all"}

	da, err := llm.StartDeviceAuth(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if da.UserCode != "WXYZ" || da.DeviceCode != "DEV" {
		t.Fatalf("device auth = %+v", da)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	tok, err := llm.PollDeviceToken(ctx, cfg, da)
	if err != nil {
		t.Fatal(err)
	}
	if tok.AccessToken != "ACCESS123" {
		t.Errorf("access token = %q", tok.AccessToken)
	}
	if atomic.LoadInt32(&polls) < 2 {
		t.Errorf("expected the pending poll to be retried, polls=%d", polls)
	}
}

func TestDeviceFlowDenied(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/device", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"device_code":"DEV","user_code":"X","verification_uri":"u","interval":1}`))
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":"access_denied"}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	cfg := llm.DeviceAuthConfig{ClientID: "cid", DeviceAuthURL: srv.URL + "/device", TokenURL: srv.URL + "/token"}
	da, err := llm.StartDeviceAuth(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := llm.PollDeviceToken(context.Background(), cfg, da); err == nil {
		t.Error("access_denied should surface as an error")
	}
}

func TestStartDeviceAuthError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":"invalid_client"}`))
	}))
	defer srv.Close()
	cfg := llm.DeviceAuthConfig{ClientID: "bad", DeviceAuthURL: srv.URL, TokenURL: srv.URL}
	if _, err := llm.StartDeviceAuth(context.Background(), cfg); err == nil {
		t.Error("invalid_client should error")
	}
}

func TestPollDeviceTokenContextCancel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":"authorization_pending"}`))
	}))
	defer srv.Close()
	cfg := llm.DeviceAuthConfig{ClientID: "c", TokenURL: srv.URL}
	da := &llm.DeviceAuth{DeviceCode: "D", Interval: 1}
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := llm.PollDeviceToken(ctx, cfg, da); err == nil {
		t.Error("expected context-cancel error while polling")
	}
}
