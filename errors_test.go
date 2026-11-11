package llm_test

import (
	"fmt"
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
