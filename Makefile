SHELL := /bin/bash

.DEFAULT_GOAL := build

.PHONY: build test test-v test-race cover test-integration vet fmt fmt-check security vulncheck check ci clean

# Build (verify compilation)
build:
	go build ./...

# Run unit tests (skips integration tests that need API keys)
test:
	go test -short -count=1 ./...

# Run tests with verbose output
test-v:
	go test -short -count=1 -v ./...

# Run tests with race detector
test-race:
	go test -short -race -count=1 ./...

# Run tests with coverage report
cover:
	go test -short -count=1 -coverpkg=./... -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# Run integration tests (requires API keys in env)
test-integration:
	go test -count=1 -v ./...

# Run go vet
vet:
	go vet ./...

# Format code
fmt:
	gofmt -w .

# Verify formatting (fails if any file needs formatting)
fmt-check:
	@test -z "$$(gofmt -l .)" || (echo "Files need formatting:"; gofmt -l .; exit 1)

# Static security analysis
# Excluded rules:
#   G117: struct fields named APIKey trigger false positive (both have MarshalJSON redaction)
#   G704: SSRF taint analysis — inherent to any HTTP client library; URLs are developer-configured
security:
	gosec -exclude=G117,G704 ./...

# Known vulnerability check
vulncheck:
	govulncheck ./...

# Quick pre-commit check
check: vet build test

# Full CI pipeline
ci: fmt-check vet build test-race security vulncheck

# Clean generated files
clean:
	rm -f coverage.out coverage.html
