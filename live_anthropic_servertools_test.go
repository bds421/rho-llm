//go:build live

package llm_test

// Live checks of the Anthropic server-tool, caching, refusal-fallback and
// thinking-replay paths against the real API. Build tag "live"; skipped when
// ANTHROPIC_API_KEY is unset. Run one at a time:
//
//	ANTHROPIC_API_KEY=... go test -tags live -run TestLiveAnthropicWebSearch -v .
//
// Cost controls: Claude Haiku 5.5 where it shows the behaviour, small
// max_tokens, at most two searches per test. Every test logs rho-llm's own
// usage and estimated cost.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
)

const (
	liveHaiku  = "claude-haiku-5-5"
	liveSonnet = "claude-sonnet-5-5"
	liveOpus   = "claude-opus-5-5"
)

func liveClient(t *testing.T, model string, maxTokens int, betas ...string) llm.Client {
	t.Helper()
	return liveClientWith(t, model, maxTokens, nil, betas...)
}

func liveClientWith(t *testing.T, model string, maxTokens int, wire func(*testing.T, *llm.Config), betas ...string) llm.Client {
	t.Helper()
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	cfg := llm.Config{
		Provider: "anthropic", Model: model, APIKey: key, MaxTokens: maxTokens,
		Timeout: 3 * time.Minute, BetaFeatures: betas,
	}
	if wire != nil {
		wire(t, &cfg)
	}
	c, err := anthropic.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func liveCtx(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	t.Cleanup(cancel)
	return ctx
}

func logResponse(t *testing.T, label string, resp *llm.Response) {
	t.Helper()
	var u llm.Usage
	u.AddResponse(resp)
	t.Logf("%s: model=%s stop=%s in=%d out=%d cacheW=%d cacheW1h=%d cacheR=%d searches=%d cost=$%.6f",
		label, resp.Model, resp.StopReason, resp.InputTokens, resp.OutputTokens,
		resp.CacheCreationTokens, resp.CacheCreation1hTokens, resp.CacheReadTokens,
		resp.ServerToolUse.WebSearchRequests, u.Cost)
	for i, it := range resp.Iterations {
		t.Logf("%s: iteration %d type=%s model=%s in=%d out=%d cacheW=%d cacheR=%d",
			label, i, it.Type, it.Model, it.InputTokens, it.OutputTokens, it.CacheCreationTokens, it.CacheReadTokens)
	}
	for i, p := range resp.Parts {
		switch p.Type {
		case llm.ContentText:
			t.Logf("%s: part %d text (%d chars, %d citations)", label, i, len(p.Text), len(p.Citations))
		case llm.ContentServerToolResult:
			errCode := ""
			if p.ServerToolError != nil {
				errCode = p.ServerToolError.ErrorCode
			}
			t.Logf("%s: part %d %s results=%d error=%q", label, i, p.BlockType, len(p.WebSearchResults), errCode)
		case llm.ContentThinking:
			t.Logf("%s: part %d thinking (%d chars, signed=%v)", label, i, len(p.Thinking), p.ThinkingSignature != "")
		default:
			t.Logf("%s: part %d %s %s", label, i, p.Type, p.ToolName+p.BlockType)
		}
	}
}

var liveSearch = llm.ServerTool{
	Type: llm.WebSearch20250305, Name: llm.ServerToolWebSearch, MaxUses: 1,
	AllowedDomains: []string{"iana.org"},
}

// 1. Web search: blocks, citations and server_tool_use usage parse, and the
// turn round-trips into the next request.
func TestLiveAnthropicWebSearch(t *testing.T) {
	c := liveClient(t, liveHaiku, 600)
	s := llm.NewSession(c, llm.WithServerTools(liveSearch))
	resp, err := s.Send(liveCtx(t), "Search the web once: what is the domain example.com reserved for, according to IANA? Answer in one sentence and cite the source.")
	if err != nil {
		t.Fatal(err)
	}
	logResponse(t, "search", resp)
	var uses, results, citations int
	for _, p := range resp.Parts {
		switch p.Type {
		case llm.ContentServerToolUse:
			uses++
		case llm.ContentServerToolResult:
			results++
		case llm.ContentText:
			citations += len(p.Citations)
		}
	}
	if uses == 0 || results == 0 || resp.ServerToolUse.WebSearchRequests == 0 {
		t.Fatalf("no search parsed: uses=%d results=%d requests=%d", uses, results, resp.ServerToolUse.WebSearchRequests)
	}
	if citations == 0 {
		t.Errorf("no citations parsed")
	}
	next, err := s.Send(liveCtx(t), "In at most five words, what was the source's site name?")
	if err != nil {
		t.Fatalf("round trip rejected: %v", err)
	}
	logResponse(t, "round-trip", next)
	t.Logf("session total cost=$%.6f searches=%d", s.Usage().Cost, s.Usage().ServerToolUse.WebSearchRequests)
}

// 3. Caching with web search: a cached system prompt, sent twice.
func TestLiveAnthropicSearchCaching(t *testing.T) {
	c := liveClient(t, liveHaiku, 300)
	system := "You answer questions about Internet domain names in one sentence.\n" +
		strings.Repeat("Reference note: example.com, example.net and example.org are reserved by IANA for documentation (RFC 2606, RFC 6761). They may be used in examples without permission. ", 120)
	req := llm.Request{
		System: system, SystemCacheControl: true,
		ServerTools: []llm.ServerTool{liveSearch},
		Messages:    []llm.Message{llm.NewTextMessage(llm.RoleUser, "Search the web once and tell me which RFC reserves example.com.")},
	}
	var total float64
	for i := 0; i < 2; i++ {
		resp, err := c.Complete(liveCtx(t), req)
		if err != nil {
			t.Fatal(err)
		}
		logResponse(t, []string{"first", "second"}[i], resp)
		var u llm.Usage
		u.AddResponse(resp)
		total += u.Cost
		if i == 1 && resp.CacheReadTokens == 0 {
			t.Errorf("second request read nothing from the cache")
		}
	}
	t.Logf("total cost=$%.6f", total)
}

// 4. max_tokens with two searches: per-response or per-iteration bound?
func TestLiveAnthropicMaxTokensAcrossSearches(t *testing.T) {
	c := liveClient(t, liveHaiku, 200)
	tool := liveSearch
	tool.MaxUses = 2
	resp, err := c.Complete(liveCtx(t), llm.Request{
		ServerTools: []llm.ServerTool{tool},
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser,
			"Do exactly two separate web searches: first for the purpose of example.com, then for the purpose of example.net. Summarise each result in two sentences.")},
	})
	if err != nil {
		t.Fatal(err)
	}
	logResponse(t, "max_tokens=200", resp)
	if resp.OutputTokens > 200 {
		t.Logf("output_tokens %d exceeds max_tokens 200: the bound is per iteration", resp.OutputTokens)
	}
}

// 5. Server-side refusal fallback: beta header plus fallbacks "default" are
// accepted (no 400).
func TestLiveAnthropicFallbackBetaAccepted(t *testing.T) {
	for _, model := range []string{liveHaiku, liveOpus} {
		t.Run(model, func(t *testing.T) {
			c := liveClient(t, model, 32, "server-side-fallback-2026-07-01")
			resp, err := c.Complete(liveCtx(t), llm.Request{
				SamplingParams: map[string]any{"fallbacks": "default"},
				Messages:       []llm.Message{llm.NewTextMessage(llm.RoleUser, "Reply with the single word: ok")},
			})
			if err != nil {
				t.Fatalf("rejected: %v", err)
			}
			logResponse(t, model, resp)
		})
	}
}

// 6. Sonnet 5.5 tool loop without a ThinkingLevel (thinking between_tools):
// its thinking blocks go back and the API accepts them.
func TestLiveAnthropicSonnetToolLoopReplaysThinking(t *testing.T) {
	c := liveClientWith(t, liveSonnet, 400, wireTypes)
	tool := llm.Tool{
		Name: "lookup_reserved_domain", Description: "Returns the registry note for a reserved example domain.",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{"domain": map[string]any{"type": "string"}}, "required": []any{"domain"}},
	}
	s := llm.NewSession(c, llm.WithTools(tool))
	resp, err := s.Send(liveCtx(t), "Call lookup_reserved_domain for example.com. Each result may name another domain to look up; follow it with one call at a time. Then answer in one sentence.")
	if err != nil {
		t.Fatal(err)
	}
	logResponse(t, "turn 1", resp)
	for round := 2; resp.StopReason == llm.StopToolUse && round <= 4; round++ {
		var results []llm.Message
		for _, tc := range resp.ToolCalls {
			note := "Reserved by IANA (RFC 2606). Registry code 48173. Next, look up the domain at index (48173 * 37) mod 5 in this list: 0 example.net, 1 example.org, 2 example.edu, 3 test.example, 4 invalid.example. Work out the index carefully before calling."
			if round > 2 {
				note = "Reserved by IANA for documentation (RFC 2606). No further lookups."
			}
			results = append(results, llm.NewToolResultMessage(tc.ID, note, false))
		}
		var merged llm.Message
		for _, m := range results {
			if merged.Role == "" {
				merged = m
				continue
			}
			merged.Content = append(merged.Content, m.Content...)
		}
		resp, err = s.SendMessages(liveCtx(t), merged)
		if err != nil {
			t.Fatalf("round %d rejected: %v", round, err)
		}
		logResponse(t, "turn "+string(rune('0'+round)), resp)
	}
	thinking := 0
	for _, m := range s.Conversation().Messages {
		for _, p := range m.Content {
			if p.Type == llm.ContentThinking {
				thinking++
			}
		}
	}
	t.Logf("thinking blocks in transcript=%d session cost=$%.6f", thinking, s.Usage().Cost)
}

// wireTypes points the client at a local reverse proxy to api.anthropic.com
// that logs the request's thinking setting and the response's block types.
func wireTypes(t *testing.T, cfg *llm.Config) {
	t.Helper()
	upstream, _ := url.Parse("https://api.anthropic.com")
	proxy := httputil.NewSingleHostReverseProxy(upstream)
	director := proxy.Director
	proxy.Director = func(r *http.Request) {
		director(r)
		r.Host = upstream.Host
		r.Header.Del("Accept-Encoding")
		body, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(body))
		var req struct {
			Thinking json.RawMessage `json:"thinking"`
			Messages []struct {
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(body, &req)
		sent := 0
		for _, m := range req.Messages {
			sent += strings.Count(string(m.Content), `"type":"thinking"`)
		}
		t.Logf("wire request: thinking=%s thinking blocks sent=%d", req.Thinking, sent)
	}
	proxy.ModifyResponse = func(resp *http.Response) error {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		resp.Body = io.NopCloser(bytes.NewReader(body))
		var out struct {
			Content []struct {
				Type string `json:"type"`
			} `json:"content"`
		}
		_ = json.Unmarshal(body, &out)
		types := make([]string, 0, len(out.Content))
		for _, c := range out.Content {
			types = append(types, c.Type)
		}
		t.Logf("wire response: status=%d blocks=%v", resp.StatusCode, types)
		return nil
	}
	srv := httptest.NewServer(proxy)
	t.Cleanup(srv.Close)
	cfg.BaseURL = srv.URL + "/v1"
}
