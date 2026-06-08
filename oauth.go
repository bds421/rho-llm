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
	status, errCode, err := postForm(ctx, cfg.DeviceAuthURL, form, &da)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		if errCode != "" {
			return nil, fmt.Errorf("llm: device authorization error: %s", errCode)
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
	interval := time.Duration(da.Interval) * time.Second
	if interval <= 0 {
		interval = 5 * time.Second
	}
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(interval):
		}
		form := url.Values{
			"client_id":   {cfg.ClientID},
			"device_code": {da.DeviceCode},
			"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
		}
		var tok Token
		status, errCode, err := postForm(ctx, cfg.TokenURL, form, &tok)
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
		case "":
			return nil, fmt.Errorf("llm: device token poll failed (status %d)", status)
		default:
			return nil, fmt.Errorf("llm: device authorization error: %s", errCode)
		}
	}
}

// postForm POSTs a urlencoded form and, on 200, decodes JSON into out; otherwise it
// best-effort parses an OAuth "error" code. Returns (status, errorCode, transportErr).
func postForm(ctx context.Context, endpoint string, form url.Values, out any) (int, string, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	resp, err := SafeHTTPClient(30 * time.Second).Do(req)
	if err != nil {
		return 0, "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode == http.StatusOK {
		if err := json.Unmarshal(data, out); err != nil {
			return resp.StatusCode, "", fmt.Errorf("llm: decode oauth response: %w", err)
		}
		return resp.StatusCode, "", nil
	}
	var e struct {
		Error string `json:"error"`
	}
	_ = json.Unmarshal(data, &e)
	return resp.StatusCode, e.Error, nil
}
