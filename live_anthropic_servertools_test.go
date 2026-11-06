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

var livePersonSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"domain":  map[string]any{"type": "string"},
		"purpose": map[string]any{"type": "string"},
	},
	"required":             []any{"domain", "purpose"},
	"additionalProperties": false,
}

// ResponseFormat as output_config.format: with adaptive thinking and effort,
// without thinking, on Sonnet 5.5 (between_tools) and on a budget model.
func TestLiveAnthropicStructuredOutput(t *testing.T) {
	cases := []struct {
		model string
		level llm.ThinkingLevel
	}{
		{liveHaiku, llm.ThinkingLow},
		{liveHaiku, llm.ThinkingNone},
		{liveSonnet, llm.ThinkingNone},
		{"claude-haiku-4-5", llm.ThinkingNone},
	}
	for _, tc := range cases {
		t.Run(tc.model+"/"+string(tc.level), func(t *testing.T) {
			c := liveClient(t, tc.model, 400)
			resp, err := c.Complete(liveCtx(t), llm.Request{
				ThinkingLevel:  tc.level,
				ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Schema: livePersonSchema},
				Messages:       []llm.Message{llm.NewTextMessage(llm.RoleUser, "What is example.com for? One short phrase.")},
			})
			if err != nil {
				t.Fatal(err)
			}
			logResponse(t, tc.model, resp)
			var out struct{ Domain, Purpose string }
			if err := json.Unmarshal([]byte(resp.Content), &out); err != nil || out.Purpose == "" {
				t.Errorf("not schema JSON: %q (%v)", resp.Content, err)
			}
		})
	}
}

// A plain-text document with title, context and a cache marker.
func TestLiveAnthropicTextDocument(t *testing.T) {
	c := liveClient(t, liveHaiku, 200)
	doc := llm.NewTextDocument("Reserved names", "The domains example.com, example.net and example.org are reserved for use in documentation (RFC 2606).")
	doc.Context = "A public registry note."
	doc.CacheControl = true
	resp, err := c.Complete(liveCtx(t), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{
		doc, {Type: llm.ContentText, Text: "According to the document, which RFC reserves these names? Answer with the RFC only."},
	}}}})
	if err != nil {
		t.Fatal(err)
	}
	logResponse(t, "text document", resp)
	if !strings.Contains(resp.Content, "2606") {
		t.Errorf("answer %q does not use the document", resp.Content)
	}
}

// display: unset sends "summarized" (readable thinking), ReasoningSummaryAuto
// leaves it off and gets the API default ("omitted": empty thinking text).
func TestLiveAnthropicThinkingDisplay(t *testing.T) {
	for _, summary := range []llm.ReasoningSummary{llm.ReasoningSummaryNone, llm.ReasoningSummaryAuto, llm.ReasoningSummaryOmitted} {
		t.Run("summary="+string(summary), func(t *testing.T) {
			c := liveClient(t, liveHaiku, 3000)
			resp, err := c.Complete(liveCtx(t), llm.Request{
				ThinkingLevel: llm.ThinkingHigh, ReasoningSummary: summary,
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "Think it through, then answer with the number only: 17 * 23 - 4 * 19.")},
			})
			if err != nil {
				t.Fatal(err)
			}
			logResponse(t, "display", resp)
			blocks := 0
			for _, p := range resp.Parts {
				if p.Type == llm.ContentThinking {
					blocks++
				}
			}
			t.Logf("thinking blocks=%d thinking chars=%d", blocks, len(resp.Thinking))
		})
	}
}

// EventDone names the answering model; CollectStream folds the stream.
func TestLiveAnthropicStreamModel(t *testing.T) {
	c := liveClient(t, liveHaiku, 32)
	resp, err := llm.CollectStream(c.Stream(liveCtx(t), llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "Reply with the single word: ok")},
	}))
	if err != nil {
		t.Fatal(err)
	}
	logResponse(t, "stream", resp)
	if resp.Model == "" || resp.InputTokens <= 0 {
		t.Errorf("stream response lacks model or usage: %+v", resp)
	}
}

// Typed Effort without thinking: Haiku 5.5 (thinking disabled), Sonnet 5.5
// (between_tools), Opus 5.5 (thinking always on), with a schema on Haiku.
func TestLiveAnthropicEffort(t *testing.T) {
	for _, tc := range []struct {
		model  string
		effort llm.Effort
		format bool
	}{
		{liveHaiku, llm.EffortLow, true},
		{liveSonnet, llm.EffortMedium, false},
		{liveOpus, llm.EffortLow, false},
	} {
		t.Run(tc.model, func(t *testing.T) {
			c := liveClient(t, tc.model, 200)
			req := llm.Request{Effort: tc.effort,
				Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "What is example.com for? One short phrase.")}}
			if tc.format {
				req.ResponseFormat = &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Schema: livePersonSchema}
			}
			resp, err := c.Complete(liveCtx(t), req)
			if err != nil {
				t.Fatalf("rejected: %v", err)
			}
			logResponse(t, tc.model, resp)
		})
	}
}

// API version 2023-06-01 pinned: temperature still reaches Haiku 4.5.
func TestLiveAnthropicPinnedOldVersion(t *testing.T) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	c, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-haiku-4-5", APIKey: key, MaxTokens: 16,
		Timeout: time.Minute, AnthropicVersion: llm.AnthropicVersion20230601})
	if err != nil {
		t.Fatal(err)
	}
	temp := 0.2
	resp, err := c.Complete(liveCtx(t), llm.Request{Temperature: &temp,
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "Reply with the single word: ok")}})
	if err != nil {
		t.Fatalf("rejected: %v", err)
	}
	logResponse(t, "pinned 2023-06-01", resp)
}

// Message Batch creation on the default version with a schema and a beta;
// waits up to four minutes for the one item, then cancels.
func TestLiveAnthropicBatch(t *testing.T) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	bc, err := llm.NewBatchClient(llm.Config{Provider: "anthropic", Model: liveHaiku, APIKey: key, Timeout: time.Minute,
		BetaFeatures: []string{"server-side-fallback-2026-07-01"}})
	if err != nil {
		t.Fatal(err)
	}
	defer bc.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	h, err := bc.Submit(ctx, []llm.BatchItem{{ItemID: "item-1", Request: &llm.Request{Model: liveHaiku, MaxTokens: 200, Effort: llm.EffortLow,
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Schema: livePersonSchema},
		Messages:       []llm.Message{llm.NewTextMessage(llm.RoleUser, "What is example.com for? One short phrase.")}}}},
		llm.BatchOptions{MaxTurnaround: 24 * time.Hour})
	if err != nil {
		t.Fatalf("batch rejected: %v", err)
	}
	t.Logf("batch %s created, status %s", h.ID, h.Status)
	deadline := time.Now().Add(4 * time.Minute)
	for time.Now().Before(deadline) {
		got, err := bc.Get(ctx, *h)
		if err != nil {
			t.Fatal(err)
		}
		if got.Status == llm.BatchCompleted {
			results, err := bc.Results(ctx, *got)
			if err != nil || len(results) != 1 || results[0].Response == nil {
				t.Fatalf("results %+v, %v", results, err)
			}
			var u llm.Usage
			u.AddBatchResponse(results[0].Response)
			t.Logf("batch result: %q in=%d out=%d cost=$%.6f", results[0].Response.Content,
				results[0].Response.InputTokens, results[0].Response.OutputTokens, u.Cost)
			return
		}
		select {
		case <-ctx.Done():
		case <-time.After(15 * time.Second):
		}
	}
	if _, err := bc.Cancel(ctx, *h); err != nil {
		t.Logf("cancel: %v", err)
	}
	t.Log("batch still running after four minutes; cancelled (creation was the check)")
}

// Streamed web search through a Session: server tool events, citations and
// usage on EventDone, and the streamed turn accepted as history.
func TestLiveAnthropicStreamedWebSearch(t *testing.T) {
	c := liveClient(t, liveHaiku, 500)
	s := llm.NewSession(c, llm.WithServerTools(liveSearch))
	var serverEvents, citations int
	for ev, err := range s.Stream(liveCtx(t), "Search the web once: what is example.org reserved for, according to IANA? One sentence.") {
		if err != nil {
			t.Fatal(err)
		}
		switch ev.Type {
		case llm.EventServerTool:
			serverEvents++
		case llm.EventCitation:
			citations++
		case llm.EventDone:
			t.Logf("done: model=%s stop=%s in=%d out=%d searches=%d parts=%d", ev.Model, ev.StopReason, ev.InputTokens, ev.OutputTokens, ev.ServerToolUse.WebSearchRequests, len(ev.Parts))
		}
	}
	t.Logf("server tool events=%d citations=%d", serverEvents, citations)
	next, err := s.Send(liveCtx(t), "In at most five words: which organisation?")
	if err != nil {
		t.Fatalf("streamed turn rejected as history: %v", err)
	}
	logResponse(t, "after stream", next)
	t.Logf("session total cost=$%.6f", s.Usage().Cost)
}
