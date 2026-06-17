package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// DeviceAuthConfig configures an OAuth 2.0 Device Authorization Grant (RFC 8628) —
// the stdlib-only way a CLI obtains an access token without a browser redirect. The
// resulting access token is used as Config.APIKey (sent as the Bearer credential),
// so this is how to wire OAuth-based providers. Endpoints, client id, and scope are
// caller-supplied (each provider documents its own).
type DeviceAuthConfig struct {
	ClientID      string
	DeviceAuthURL string // device_authorization endpoint
	TokenURL      string // token endpoint
	Scope         string // optional
}

// DeviceAuth is the device-authorization response. Show UserCode and
// VerificationURI to the user, then call PollDeviceToken.
type DeviceAuth struct {
	DeviceCode              string `json:"device_code"`
	UserCode                string `json:"user_code"`
	VerificationURI         string `json:"verification_uri"`
	VerificationURIComplete string `json:"verification_uri_complete"`
	ExpiresIn               int    `json:"expires_in"`
	Interval                int    `json:"interval"`
}

// Token is an OAuth token response.
type Token struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
}

// StartDeviceAuth requests a device + user code from the device_authorization endpoint.
func StartDeviceAuth(ctx context.Context, cfg DeviceAuthConfig) (*DeviceAuth, error) {
	form := url.Values{"client_id": {cfg.ClientID}}
	if cfg.Scope != "" {
		form.Set("scope", cfg.Scope)
	}
	var da DeviceAuth
	status, errCode, errDesc, err := postForm(ctx, cfg.DeviceAuthURL, form, &da)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		if errCode != "" {
			return nil, fmt.Errorf("llm: device authorization error: %s", oauthErrMsg(errCode, errDesc))
		}
		return nil, fmt.Errorf("llm: device authorization failed (status %d)", status)
	}
	if da.DeviceCode == "" {
		return nil, fmt.Errorf("llm: device authorization returned no device_code")
	}
	if da.Interval <= 0 {
		da.Interval = 5
	}
	return &da, nil
}

// PollDeviceToken polls the token endpoint until the user authorizes, the grant
// expires, or ctx is cancelled. It honors the RFC 8628 "authorization_pending" and
// "slow_down" responses.
func PollDeviceToken(ctx context.Context, cfg DeviceAuthConfig, da *DeviceAuth) (*Token, error) {
	interval := clampPollInterval(da.Interval)
	// Self-imposed expiry: stop polling once the device code's lifetime elapses
	// (RFC 8628). Without this the loop relies entirely on the server returning
	// expired_token + the caller's ctx — a server stuck on authorization_pending
	// with no ctx deadline would poll forever.
	var deadline time.Time
	if da.ExpiresIn > 0 {
		deadline = time.Now().Add(clampExpiry(da.ExpiresIn))
	}
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(interval):
		}
		if !deadline.IsZero() && time.Now().After(deadline) {
			return nil, fmt.Errorf("llm: device code expired before authorization (expires_in=%ds)", da.ExpiresIn)
		}
		form := url.Values{
			"client_id":   {cfg.ClientID},
			"device_code": {da.DeviceCode},
			"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
		}
		var tok Token
		status, errCode, errDesc, err := postForm(ctx, cfg.TokenURL, form, &tok)
		if err != nil {
			return nil, err
		}
		if status == http.StatusOK && tok.AccessToken != "" {
			return &tok, nil
		}
		switch errCode {
		case "authorization_pending":
			continue
		case "slow_down":
			interval += 5 * time.Second
			if interval > maxPollInterval {
				interval = maxPollInterval
			}
		case "":
			return nil, fmt.Errorf("llm: device token poll failed (status %d)", status)
		default:
			return nil, fmt.Errorf("llm: device authorization error: %s", oauthErrMsg(errCode, errDesc))
		}
	}
}

// poll-interval bounds. RFC 8628 leaves the interval to the server; we enforce a
// sane floor (avoid hammering) and ceiling (a huge/overflowing value must not wrap
// to a tiny — or negative — duration that defeats the floor).
const (
	defaultPollInterval = 5 * time.Second
	maxPollInterval     = 300 * time.Second
	maxDeviceExpiry     = 24 * 60 * 60 // seconds; caps ExpiresIn so the deadline math can't overflow
)

// clampPollInterval converts a server-supplied interval (seconds) to a duration
// bounded to [1s, maxPollInterval], defaulting a non-positive value. Clamping the
// int BEFORE multiplying keeps the result overflow-safe: a hostile interval like
// math.MaxInt can no longer wrap past int64 into a tiny/negative poll delay.
func clampPollInterval(seconds int) time.Duration {
	if seconds <= 0 {
		return defaultPollInterval
	}
	if seconds > int(maxPollInterval/time.Second) {
		return maxPollInterval
	}
	return time.Duration(seconds) * time.Second
}

// clampExpiry bounds ExpiresIn (seconds) so the polling deadline can't overflow.
func clampExpiry(seconds int) time.Duration {
	if seconds > maxDeviceExpiry {
		seconds = maxDeviceExpiry
	}
	return time.Duration(seconds) * time.Second
}

// oauthErrMsg combines an OAuth error code with its (optional) RFC-mandated
// human-readable error_description.
func oauthErrMsg(code, desc string) string {
	if desc != "" {
		return code + ": " + desc
	}
	return code
}

// postForm POSTs a urlencoded form and, on 200, decodes JSON into out; otherwise it
// best-effort parses the OAuth "error" code and "error_description".
// Returns (status, errorCode, errorDescription, transportErr).
func postForm(ctx context.Context, endpoint string, form url.Values, out any) (int, string, string, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return 0, "", "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	resp, err := SafeHTTPClient(30 * time.Second).Do(req)
	if err != nil {
		return 0, "", "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode == http.StatusOK {
		if err := json.Unmarshal(data, out); err != nil {
			return resp.StatusCode, "", "", fmt.Errorf("llm: decode oauth response: %w", err)
		}
		return resp.StatusCode, "", "", nil
	}
	var e struct {
		Error            string `json:"error"`
		ErrorDescription string `json:"error_description"`
	}
	_ = json.Unmarshal(data, &e)
	return resp.StatusCode, e.Error, e.ErrorDescription, nil
}
