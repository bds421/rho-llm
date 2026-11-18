package llm

import (
	"errors"
	"regexp"
)

// Anthropic message-thread error codes (error.details.error_code on a 400 or
// 404), see ThreadErrorCode.
const (
	ThreadNotFound         = "thread_not_found"
	ThreadFingerprint      = "thread_fingerprint_mismatch"
	ThreadAlreadyContinued = "thread_already_continued"
	ThreadContinueInFlight = "thread_continue_in_flight"
	ThreadUnsupported      = "thread_unsupported_request"
)

var threadErrorCode = regexp.MustCompile(`"error_code"\s*:\s*"(thread_[a-z_]+)"`)

// ThreadErrorCode returns the message-thread error code (ThreadNotFound, …)
// an Anthropic 400 or 404 carries, or "" when err is not one: any other
// error, status or body. The code is read from the body, whose text may be
// truncated, so a body without it yields "" rather than a guess.
func ThreadErrorCode(err error) string {
	var apiErr *APIError
	if !errors.As(err, &apiErr) || (apiErr.StatusCode != 400 && apiErr.StatusCode != 404) {
		return ""
	}
	if m := threadErrorCode.FindStringSubmatch(apiErr.Message); m != nil {
		return m[1]
	}
	return ""
}
