package openaibatch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"

	"github.com/bds421/rho-llm"
)

// setAuth applies the bearer (or custom) authorization header. No-op when the provider
// needs no auth or no key is configured.
func (c *Client) setAuth(req *http.Request) {
	if c.authHeader != "" && c.cfg.APIKey != "" {
		req.Header.Set("Authorization", c.authHeader+" "+c.cfg.APIKey)
	}
}

// doJSON issues an HTTP request with an optional JSON body and decodes a JSON response
// into out (out may be nil). Non-2xx becomes a classified llm.APIError via the shared
// transport helper (TLS, redirect auth-stripping, key redaction all inherited).
func (c *Client) doJSON(ctx context.Context, method, endpoint string, body, out any) error {
	return c.doJSONWithConfig(ctx, c.cfg, method, endpoint, body, out)
}

func (c *Client) doJSONSingle(ctx context.Context, method, endpoint string, body, out any) error {
	cfg := c.cfg
	cfg.DisableRetries = true
	return c.doJSONWithConfig(ctx, cfg, method, endpoint, body, out)
}

func (c *Client) doJSONWithConfig(
	ctx context.Context,
	cfg llm.Config,
	method, endpoint string,
	body, out any,
) error {
	var data []byte
	if body != nil {
		var err error
		data, err = json.Marshal(body)
		if err != nil {
			return err
		}
	}
	resp, err := llm.DoHTTP(ctx, cfg, c.httpClient, func(ctx context.Context) (*http.Request, error) {
		var rdr io.Reader
		if body != nil {
			rdr = bytes.NewReader(data)
		}
		req, buildErr := http.NewRequestWithContext(ctx, method, endpoint, rdr)
		if buildErr != nil {
			return nil, fmt.Errorf("openaibatch: build request: %w", buildErr)
		}
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		c.setAuth(req)
		return req, nil
	})
	if err != nil {
		return fmt.Errorf("openaibatch: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		return llm.ErrorFromResponse(c.providerName, resp, cfg)
	}
	if out != nil {
		if err := llm.DecodeJSONResponse(resp, cfg, out); err != nil {
			return fmt.Errorf("openaibatch: decode response: %w", err)
		}
	}
	return nil
}

// uploadInputFile uploads the assembled JSONL as a multipart file with purpose=batch
// and returns the new file id. Mirrors the multipart pattern in the
// OpenAI-compatible modality adapter's TranscribeAudio.
func (c *Client) uploadInputFile(ctx context.Context, jsonl []byte) (string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	if err := mw.WriteField("purpose", "batch"); err != nil {
		return "", err
	}
	fw, err := mw.CreateFormFile("file", "batch.jsonl")
	if err != nil {
		return "", err
	}
	if _, err := fw.Write(jsonl); err != nil {
		return "", err
	}
	if err := mw.Close(); err != nil {
		return "", err
	}

	payload := append([]byte(nil), buf.Bytes()...)
	contentType := mw.FormDataContentType()
	cfg := c.cfg
	cfg.DisableRetries = true
	resp, err := llm.DoHTTP(ctx, cfg, c.httpClient, func(ctx context.Context) (*http.Request, error) {
		req, buildErr := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/files", bytes.NewReader(payload))
		if buildErr != nil {
			return nil, fmt.Errorf("openaibatch: build file upload: %w", buildErr)
		}
		req.Header.Set("Content-Type", contentType)
		c.setAuth(req)
		return req, nil
	})
	if err != nil {
		return "", fmt.Errorf("openaibatch: file upload failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		return "", llm.ErrorFromResponse(c.providerName, resp, cfg)
	}
	var out struct {
		ID string `json:"id"`
	}
	if err := llm.DecodeJSONResponse(resp, cfg, &out); err != nil {
		return "", fmt.Errorf("openaibatch: decode file upload response: %w", err)
	}
	if out.ID == "" {
		return "", fmt.Errorf("openaibatch: file upload returned no id")
	}
	return out.ID, nil
}

// downloadFile fetches a result/error file's content, bounded by
// EffectiveMaxBatchDownloadBytes. A read that hits the cap is treated as oversize
// (and errored) rather than silently truncated; a mid-read failure is surfaced, not
// returned as a partial body.
func (c *Client) downloadFile(ctx context.Context, fileID string) ([]byte, error) {
	endpoint := c.baseURL + "/files/" + url.PathEscape(fileID) + "/content"
	resp, err := llm.DoHTTP(ctx, c.cfg, c.httpClient, func(ctx context.Context) (*http.Request, error) {
		req, buildErr := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
		if buildErr != nil {
			return nil, fmt.Errorf("openaibatch: build download: %w", buildErr)
		}
		c.setAuth(req)
		return req, nil
	})
	if err != nil {
		return nil, fmt.Errorf("openaibatch: download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		return nil, llm.ErrorFromResponse(c.providerName, resp, c.cfg)
	}

	limit := c.cfg.EffectiveMaxBatchDownloadBytes()
	// Read one byte past the cap so an exactly-cap-sized file still succeeds while
	// anything larger is detected rather than silently truncated.
	data, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, fmt.Errorf("openaibatch: incomplete batch download from %s: %w", c.providerName, err)
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("openaibatch: batch result file exceeds %d bytes (raise Config.MaxBatchDownloadBytes)", limit)
	}
	return data, nil
}
