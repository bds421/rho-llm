package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

type failingBody struct{}

func (failingBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (failingBody) Close() error             { return nil }

func TestCreateCacheRequiresDeadlineAndName(t *testing.T) {
	original := cacheHTTPClient
	t.Cleanup(func() { cacheHTTPClient = original })

	cacheHTTPClient = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", request.Method)
		}
		if _, ok := request.Context().Deadline(); !ok {
			t.Fatal("cache request has no context deadline")
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"name":"cachedContents/test"}`)),
			Header:     make(http.Header),
		}, nil
	})}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	name, err := createCache(ctx, "test-key", "large prompt")
	if err != nil || name != "cachedContents/test" {
		t.Fatalf("createCache() = %q, %v", name, err)
	}

	cacheHTTPClient.Transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{}`)),
			Header:     make(http.Header),
		}, nil
	})
	if _, err := createCache(ctx, "test-key", "large prompt"); err == nil {
		t.Fatal("createCache accepted a response without a cache name")
	}

	cacheHTTPClient.Transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: failingBody{}, Header: make(http.Header)}, nil
	})
	if _, err := createCache(ctx, "test-key", "large prompt"); err == nil {
		t.Fatal("createCache ignored a response body read failure")
	}
}

func TestDeleteCacheUsesCallerContext(t *testing.T) {
	original := cacheHTTPClient
	t.Cleanup(func() { cacheHTTPClient = original })
	cacheHTTPClient = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.Method != http.MethodDelete {
			t.Fatalf("method = %s, want DELETE", request.Method)
		}
		if _, ok := request.Context().Deadline(); !ok {
			t.Fatal("delete request has no context deadline")
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{}`)),
			Header:     make(http.Header),
		}, nil
	})}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := deleteCache(ctx, "test-key", "cachedContents/test"); err != nil {
		t.Fatal(err)
	}
}
