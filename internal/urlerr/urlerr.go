// Package urlerr scrubs credentials from the request URL that net/http puts
// into every transport error.
package urlerr

import (
	"errors"
	"net/url"
	"strings"
)

// sensitiveQueryKeys are query parameters that carry a credential.
var sensitiveQueryKeys = map[string]bool{
	"key": true, "api_key": true, "apikey": true, "token": true, "access_token": true, "auth": true,
}

// Redact returns err with the URL of its *url.Error scrubbed: userinfo (a key
// carried as the username survived, net/http strips only the password) and
// credential query parameters. The error keeps its type and wrapped cause, so
// retry classification is unchanged. Other errors are returned as is.
func Redact(err error) error {
	var ue *url.Error
	if !errors.As(err, &ue) {
		return err
	}
	clean := redactURL(ue.URL)
	if clean == ue.URL {
		return err
	}
	if err == error(ue) {
		return &url.Error{Op: ue.Op, URL: clean, Err: ue.Err}
	}
	// Wrapped deeper: keep the chain, replace only the text.
	return &redacted{msg: strings.ReplaceAll(err.Error(), ue.URL, clean), err: &url.Error{Op: ue.Op, URL: clean, Err: ue.Err}}
}

type redacted struct {
	msg string
	err error
}

func (r *redacted) Error() string { return r.msg }
func (r *redacted) Unwrap() error { return r.err }

func redactURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	changed := false
	if u.User != nil {
		u.User = url.User("REDACTED")
		changed = true
	}
	q := u.Query()
	for k := range q {
		if sensitiveQueryKeys[strings.ToLower(k)] {
			q.Set(k, "REDACTED")
			changed = true
		}
	}
	if !changed {
		return raw
	}
	u.RawQuery = q.Encode()
	return u.String()
}
