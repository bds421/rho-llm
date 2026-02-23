# Security Audit & Hardening — rho/llm

## Plan

- [x] Add `gosec` and `govulncheck` to Makefile CI
- [x] Enforce TLS 1.2+ minimum in `SafeHTTPClient`
- [x] Add security test: `Config.MarshalJSON` redacts API keys
- [x] Add security test: `AuthProfile.MarshalJSON` redacts API keys
- [x] Add security test: BaseURL scheme validation (SSRF boundary)
- [x] Run full verification: tests, gosec, govulncheck, go vet

## Changes Summary

### `Makefile`
- Added `security` target: `gosec -exclude=G117,G704 ./...`
- Added `vulncheck` target: `govulncheck ./...`
- Both added to `ci` target

### `config.go`
- Added `crypto/tls` import
- Set `TLSClientConfig: &tls.Config{MinVersion: tls.VersionTLS12}` on the `http.Transport` inside `SafeHTTPClient`

### `backoff.go`
- Added `#nosec G404` annotation to backoff jitter (math/rand is correct here)

### `go.mod`
- Upgraded from `go 1.23.4` to `go 1.26.0`

### `llm_test.go`
- Fixed `fmt.Errorf` non-constant format string (Go 1.26 stricter vet)

### `README.md`
- Updated Go version requirement from 1.23+ to 1.26+

### `security_test.go`
- `TestConfigMarshalJSONRedactsAPIKey` — verifies `Config.MarshalJSON()` replaces APIKey with "REDACTED"
- `TestAuthProfileMarshalJSONRedactsAPIKey` — verifies `AuthProfile.MarshalJSON()` redacts
- `TestBaseURLSchemeValidation` — verifies http/https work, file/javascript/ftp/data schemes error clearly

## Verification Results

| Check | Result |
|-------|--------|
| `go test -short -race -count=1 ./...` | PASS (41s) |
| `go vet ./...` | Clean |
| `gosec -exclude=G117,G704 ./...` | 0 issues (1 nosec) |
| `govulncheck ./...` | No vulnerabilities found (Go 1.26.0) |

### gosec Exclusion Justifications

- **G117** (secret pattern): `Config.APIKey` and `AuthProfile.APIKey` both have `MarshalJSON` that redacts to "REDACTED"
- **G704** (SSRF taint): Inherent to any HTTP client library — the library is designed to make requests to developer-configured URLs
- **G404** (weak RNG): `#nosec` on backoff jitter — crypto randomness is unnecessary for retry delay
