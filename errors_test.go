package llm_test

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
	_ "github.com/bds421/rho-llm/provider"
)

// TestAPIErrorTypes verifies error type construction and classification.
func TestAPIErrorTypes(t *testing.T) {
	tests := []struct {
		name          string
		err           *llm.APIError
		isRateLimited bool
		isOverloaded  bool
		isRetryable   bool
		isAuth        bool
		isCtxLength   bool
	}{
		{
			name:          "rate limit",
			err:           llm.NewRateLimitError("anthropic", "rate limit exceeded"),
			isRateLimited: true,
			isRetryable:   true,
		},
		{
			name:         "overloaded",
			err:          llm.NewOverloadedError("gemini", "service unavailable"),
			isOverloaded: true,
			isRetryable:  true,
		},
		{
			name:   "auth error 401",
			err:    llm.NewAuthError("xai", "unauthorized", 401),
			isAuth: true,
		},
		{
			name:   "auth error 403",
			err:    llm.NewAuthError("openai", "forbidden", 403),
			isAuth: true,
		},
		{
			name:        "context length",
			err:         llm.NewContextLengthError("anthropic", "maximum context length exceeded"),
			isCtxLength: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if llm.IsRateLimited(tc.err) != tc.isRateLimited {
				t.Errorf("IsRateLimited = %v, want %v", llm.IsRateLimited(tc.err), tc.isRateLimited)
			}
			if llm.IsOverloaded(tc.err) != tc.isOverloaded {
				t.Errorf("IsOverloaded = %v, want %v", llm.IsOverloaded(tc.err), tc.isOverloaded)
			}
			if llm.IsRetryable(tc.err) != tc.isRetryable {
				t.Errorf("IsRetryable = %v, want %v", llm.IsRetryable(tc.err), tc.isRetryable)
			}
			if llm.IsAuthError(tc.err) != tc.isAuth {
				t.Errorf("IsAuthError = %v, want %v", llm.IsAuthError(tc.err), tc.isAuth)
			}
			if llm.IsContextLength(tc.err) != tc.isCtxLength {
				t.Errorf("IsContextLength = %v, want %v", llm.IsContextLength(tc.err), tc.isCtxLength)
			}
		})
	}
}

// TestAPIErrorFromStatus verifies automatic error classification from HTTP status.
func TestAPIErrorFromStatus(t *testing.T) {
	tests := []struct {
		status      int
		body        string
		isRetryable bool
		isAuth      bool
		isCtxLen    bool
	}{
		{429, "rate limit exceeded", true, false, false},
		{503, "overloaded", true, false, false},
		{401, "invalid api key", false, true, false},
		{403, "forbidden", false, true, false},
		{400, "maximum context length exceeded", false, false, true},
		{400, "context_length_exceeded: max 8192 tokens", false, false, true},
		{400, "input too long for model", false, false, true},
		{400, "request too large", false, false, true},
		{400, `{"error":{"code":400,"message":"The input token count exceeds the maximum number of tokens allowed 1048576.","status":"INVALID_ARGUMENT"}}`, false, false, true},
		{400, "bad request", false, false, false},
		{500, "internal server error", true, false, false},
	}

	for _, tc := range tests {
		name := fmt.Sprintf("status_%d", tc.status)
		t.Run(name, func(t *testing.T) {
			err := llm.NewAPIErrorFromStatus("test", tc.status, tc.body)
			if llm.IsRetryable(err) != tc.isRetryable {
				t.Errorf("IsRetryable = %v, want %v", llm.IsRetryable(err), tc.isRetryable)
			}
			if llm.IsAuthError(err) != tc.isAuth {
				t.Errorf("IsAuthError = %v, want %v", llm.IsAuthError(err), tc.isAuth)
			}
			if llm.IsContextLength(err) != tc.isCtxLen {
				t.Errorf("IsContextLength = %v, want %v", llm.IsContextLength(err), tc.isCtxLen)
			}
		})
	}
}

// TestAPIErrorNilSafety verifies nil errors don't panic.
func TestAPIErrorNilSafety(t *testing.T) {
	if llm.IsRateLimited(nil) {
		t.Error("nil should not be rate limited")
	}
	if llm.IsRetryable(nil) {
		t.Error("nil should not be retryable")
	}
	if llm.IsAuthError(nil) {
		t.Error("nil should not be auth error")
	}
}

// TestIsOverloadedFalse verifies IsOverloaded returns false for non-503 errors.
func TestIsOverloadedFalse(t *testing.T) {
	err := llm.NewRateLimitError("test", "rate limited")
	if llm.IsOverloaded(err) {
		t.Error("IsOverloaded should return false for 429 error")
	}

	// Non-APIError
	plainErr := fmt.Errorf("some error")
	if llm.IsOverloaded(plainErr) {
		t.Error("IsOverloaded should return false for non-APIError")
	}

	// nil
	if llm.IsOverloaded(nil) {
		t.Error("IsOverloaded should return false for nil")
	}
}

// TestIsContextLengthFalse verifies IsContextLength returns false for non-matching errors.
func TestIsContextLengthFalse(t *testing.T) {
	// 400 but not context length message
	err := llm.NewAPIErrorFromStatus("test", 400, "invalid json")
	if llm.IsContextLength(err) {
		t.Error("IsContextLength should return false for non-context-length 400")
	}

	// Non-400
	err = llm.NewRateLimitError("test", "rate limited")
	if llm.IsContextLength(err) {
		t.Error("IsContextLength should return false for 429")
	}

	// nil
	if llm.IsContextLength(nil) {
		t.Error("IsContextLength should return false for nil")
	}
}

// TestContextLengthFalsePositives verifies rate-limit-like messages are NOT classified as context length.
func TestContextLengthFalsePositives(t *testing.T) {
	falsePositives := []string{
		"token limit temporarily reached",
		"per token limit exceeded",
		"too many tokens in this request batch",
	}

	for _, msg := range falsePositives {
		t.Run(msg, func(t *testing.T) {
			err := llm.NewAPIErrorFromStatus("test", 400, msg)
			if llm.IsContextLength(err) {
				t.Errorf("IsContextLength(%q) = true, should be false (false positive)", msg)
			}
		})
	}

	// Also verify true positives still work
	truePositives := []string{
		"context length exceeded",
		"maximum context window exceeded",
		"prompt is too long",
		"input too long for model",
		"request too large",
		"input token count exceeds the maximum",
	}

	for _, msg := range truePositives {
		t.Run("true_"+msg, func(t *testing.T) {
			err := llm.NewAPIErrorFromStatus("test", 400, msg)
			if !llm.IsContextLength(err) {
				t.Errorf("IsContextLength(%q) = false, should be true", msg)
			}
		})
	}
}

func TestOverloadedStatuses(t *testing.T) {
	for _, status := range []int{503, 529} {
		err := llm.NewAPIErrorFromStatus("anthropic", status, "overloaded")
		if !llm.IsRetryable(err) || !llm.IsOverloaded(err) || err.StatusCode != status {
			t.Errorf("%d: %+v", status, err)
		}
	}
	if !llm.IsRetryable(llm.NewAPIErrorFromStatus("anthropic", 504, "timeout")) {
		t.Error("504 not retryable")
	}
}

// Overflow messages from many hosts were missed (only seven phrases, status
// 400 only), and a 413 never counted (P5). Rate-limit wording stays out.
func TestContextLengthAcrossHosts(t *testing.T) {
	overflow := map[string]struct {
		status int
		body   string
	}{
		"xAI":           {400, "This model's maximum prompt length is 131072 but the request contains 537812 tokens"},
		"Groq":          {400, "Please reduce the length of the messages or completion"},
		"Together":      {400, "The input (300000 tokens) is longer than the model's context length (262144 tokens)."},
		"llama.cpp":     {400, "the request exceeds the available context size, try increasing it"},
		"LM Studio":     {400, "tokens to keep from the initial prompt is greater than the context length"},
		"MiniMax":       {400, "invalid params, context window exceeds limit"},
		"Kimi":          {400, "Your request exceeded model token limit: 262144 (requested: 300000)"},
		"DashScope":     {400, "Range of input length should be [1, 129024]"},
		"Z.ai":          {400, `{"code":"1261","message":"Prompt too long"}`},
		"OpenAI":        {400, "Your input exceeds the context window of this model"},
		"422":           {422, "maximum context length is 8192 tokens"},
		"Anthropic 413": {413, `{"type":"error","error":{"type":"request_too_large","message":"Request exceeds the maximum size"}}`},
		"413 no body":   {413, ""},
	}
	for name, tc := range overflow {
		err := llm.NewAPIErrorFromStatus("p", tc.status, tc.body)
		if !llm.IsContextLength(err) || llm.IsRetryable(err) {
			t.Errorf("%s: IsContextLength=%v IsRetryable=%v, want true/false", name, llm.IsContextLength(err), llm.IsRetryable(err))
		}
	}
	notOverflow := map[string]struct {
		status int
		body   string
	}{
		"Bedrock throttle":       {400, "ThrottlingException: Too many tokens, please wait before trying again."},
		"429 with overflow text": {429, "context length exceeded for this minute"},
		"rate limit wording":     {400, "rate limit: maximum context tokens per minute"},
		"500":                    {500, "maximum context length is 8192 tokens"},
		"plain 400":              {400, "invalid tool schema"},
	}
	for name, tc := range notOverflow {
		if err := llm.NewAPIErrorFromStatus("p", tc.status, tc.body); llm.IsContextLength(err) {
			t.Errorf("%s: classified as context overflow", name)
		}
	}
	// A stream error carries no HTTP status; its text still counts.
	if !llm.IsContextLength(&llm.APIError{Provider: "p", Message: "prompt is too long: 213462 tokens > 200000 maximum"}) {
		t.Error("status-less overflow error missed")
	}
}

// An exhausted quota was retried like a throttle; it resets in hours (P5).
// Gemini's ordinary per-minute 429 mentions "billing" and must stay retryable.
func TestQuotaExhaustedIsNotRetried(t *testing.T) {
	for _, body := range []string{
		`{"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota","code":"insufficient_quota"}}`,
		`{"error":{"code":"1308","message":"Usage limit reached for 5 hour"}}`,
		`{"error":"Monthly usage limit reached"}`,
		"Quota exceeded for this project",
	} {
		err := llm.NewAPIErrorFromStatus("p", 429, body)
		if llm.IsRetryable(err) || !llm.IsQuotaExhausted(err) || !llm.IsRateLimited(err) {
			t.Errorf("%s: retryable=%v quota=%v", body, llm.IsRetryable(err), llm.IsQuotaExhausted(err))
		}
	}
	for _, body := range []string{
		"RESOURCE_EXHAUSTED: You exceeded your current quota, please check your plan and billing details. Please retry in 21s.",
		`{"error":{"code":"13080","message":"busy"}}`,
		"Rate limit reached for requests",
	} {
		err := llm.NewAPIErrorFromStatus("p", 429, body)
		if !llm.IsRetryable(err) || llm.IsQuotaExhausted(err) {
			t.Errorf("%s: retryable=%v quota=%v, want a plain throttle", body, llm.IsRetryable(err), llm.IsQuotaExhausted(err))
		}
	}
	if llm.IsQuotaExhausted(nil) || llm.IsQuotaExhausted(errors.New("insufficient_quota")) {
		t.Error("only an APIError can be quota-exhausted")
	}
}

// The provider's retry headers override the status rule (P5).
func TestErrorFromResponseHonoursRetryHeaders(t *testing.T) {
	mk := func(status int, h map[string]string) error {
		resp := &http.Response{StatusCode: status, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("x"))}
		for k, v := range h {
			resp.Header.Set(k, v)
		}
		return llm.ErrorFromResponse("p", resp, llm.Config{})
	}
	if err := mk(503, map[string]string{"X-Should-Retry": "false"}); llm.IsRetryable(err) {
		t.Error("x-should-retry: false ignored")
	}
	if err := mk(409, map[string]string{"x-should-retry": " TRUE "}); !llm.IsRetryable(err) {
		t.Error("x-should-retry: true ignored")
	}
	err := mk(429, map[string]string{"Retry-After": "3600"})
	if llm.IsRetryable(err) || !llm.IsQuotaExhausted(err) {
		t.Errorf("Retry-After 3600: retryable=%v quota=%v", llm.IsRetryable(err), llm.IsQuotaExhausted(err))
	}
	if err := mk(429, map[string]string{"Retry-After": "60"}); !llm.IsRetryable(err) || llm.IsQuotaExhausted(err) {
		t.Error("Retry-After at the cap must stay retryable")
	}
	if err := mk(429, map[string]string{"Retry-After": "99999999999999999999"}); llm.IsRetryable(err) {
		t.Error("overflowing Retry-After must not be retried")
	}
}
