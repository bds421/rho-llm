package anthropic

import llm "github.com/bds421/rho-llm"

// streamErrorStatus maps the error types of a mid-stream "error" event onto
// the HTTP status the same error carries outside a stream, so retry and
// failover classify it alike.
var streamErrorStatus = map[string]int{
	"invalid_request_error": 400,
	"authentication_error":  401,
	"billing_error":         402,
	"permission_error":      403,
	"not_found_error":       404,
	"request_too_large":     413,
	"rate_limit_error":      429,
	"api_error":             500,
	"timeout_error":         504,
	"overloaded_error":      529,
}

// streamError builds the error for a mid-stream "error" event. The message
// gets the same hygiene as an HTTP error body: the provider may echo the
// request key, so the deployment's secrets are scrubbed first.
func streamError(errType, message string, cfg llm.Config) error {
	status, ok := streamErrorStatus[errType]
	if !ok {
		status = 500
	}
	message = llm.RedactSecrets(message, cfg)
	return llm.NewAPIErrorFromStatusWithLimit("anthropic", status, errType+": "+message, cfg.EffectiveMaxErrorMessageLen())
}
