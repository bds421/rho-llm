package llm_test

// C5 — cache-token double charge. Break-the-system tests for the disjoint
// cache-token contract: Response.InputTokens is the UNCACHED input only and
// CacheReadTokens is separate, so EstimateCost prices every prompt token once.
//
// Wire facts (primary sources):
//   - Gemini: promptTokenCount "includes the number of tokens in the cached
//     content" (ai.google.dev/api/generate-content, UsageMetadata).
//   - OpenAI Chat Completions: prompt_tokens_details.cached_tokens = "Cached
//     tokens present in the prompt"; Responses: input_tokens includes
//     input_tokens_details.cached_tokens (developers.openai.com prompt-caching
//     guide: ordinary input = input_tokens - cached_tokens - cache_write_tokens).
//   - Anthropic: total_input_tokens = cache_read_input_tokens +
//     cache_creation_input_tokens + input_tokens (platform.claude.com
//     prompt-caching) — already disjoint.
//
// Before the fix the Gemini adapter passed promptTokenCount through as
// InputTokens AND cachedContentTokenCount as CacheReadTokens (cached tokens
// charged at the input price plus again at the cache price), and the
// OpenAI-style adapters ignored cached_tokens entirely.

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
	"github.com/bds421/rho-llm/provider/gemini"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

const (
	c5InputPrice     = 10.0 // USD / 1M uncached input
	c5CacheReadPrice = 1.0  // USD / 1M cache-read input
)

// c5Model registers a model whose output is free, so cost isolates input
// pricing: uncached × 10 + cached × 1 (per 1M).
func c5Model(t *testing.T) string {
	t.Helper()
	id := uniqueModelID("test-only-c5-cache")
	if err := llm.RegisterModel(llm.ModelInfo{
		ID: id, Provider: "custom",
		InputPricePer1M: c5InputPrice, CacheReadPricePer1M: c5CacheReadPrice,
	}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	return id
}

func c5WantCost(uncached, cached int) float64 {
	return float64(uncached)*c5InputPrice/1e6 + float64(cached)*c5CacheReadPrice/1e6
}

func jsonServer(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}))
}

// c5Wire builds one adapter's non-streaming body and streaming SSE lines for
// a usage object rendered by usageJSON. usage may be "" (field absent).
type c5Adapter struct {
	name      string
	newClient func(baseURL string) (llm.Client, error)
	complete  func(usage string) string
	stream    func(usage string) []string
	// wire renders the SAME logical usage (uncached, cached) in this
	// provider's own shape.
	wire func(uncached, cached int) string
}

func c5Adapters() []c5Adapter {
	cfg := func(provider, model, base string) llm.Config {
		return llm.Config{Provider: provider, Model: model, APIKey: "test-key", BaseURL: base, Timeout: 10 * time.Second}
	}
	return []c5Adapter{
		{
			name: "anthropic",
			newClient: func(b string) (llm.Client, error) {
				return anthropic.New(cfg("anthropic", "claude-sonnet-4-6", b))
			},
			complete: func(u string) string {
				return `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn"` + c5Field("usage", u) + `}`
			},
			stream: func(u string) []string {
				return []string{
					`data: {"type":"message_start","message":{"id":"msg_1"` + c5Field("usage", u) + `}}`,
					`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
					`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
					`data: {"type":"content_block_stop","index":0}`,
					`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}`,
					`data: {"type":"message_stop"}`,
				}
			},
			wire: func(un, ca int) string {
				return fmt.Sprintf(`{"input_tokens":%d,"output_tokens":0,"cache_read_input_tokens":%d}`, un, ca)
			},
		},
		{
			name: "gemini",
			newClient: func(b string) (llm.Client, error) {
				return gemini.New(cfg("gemini", "gemini-2.5-flash", b))
			},
			complete: func(u string) string {
				return `{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]` + c5Field("usageMetadata", u) + `}`
			},
			stream: func(u string) []string {
				return []string{`data: {"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]` + c5Field("usageMetadata", u) + `}`}
			},
			wire: func(un, ca int) string {
				return fmt.Sprintf(`{"promptTokenCount":%d,"candidatesTokenCount":0,"cachedContentTokenCount":%d}`, un+ca, ca)
			},
		},
		{
			name: "openai_compat",
			newClient: func(b string) (llm.Client, error) {
				return openaicompat.New(cfg("openai", "gpt-4.1", b))
			},
			complete: func(u string) string {
				return `{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]` + c5Field("usage", u) + `}`
			},
			stream: func(u string) []string {
				return []string{
					`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
					`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
					`data: {"id":"c1","choices":[]` + c5Field("usage", u) + `}`,
					`data: [DONE]`,
				}
			},
			wire: func(un, ca int) string {
				return fmt.Sprintf(`{"prompt_tokens":%d,"completion_tokens":0,"prompt_tokens_details":{"cached_tokens":%d}}`, un+ca, ca)
			},
		},
		{
			name: "openai_responses",
			newClient: func(b string) (llm.Client, error) {
				return openairesponses.New(cfg("openai_responses", "gpt-5.2", b))
			},
			complete: func(u string) string {
				return `{"id":"resp_1","model":"gpt-5.2","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}]` + c5Field("usage", u) + `}`
			},
			stream: func(u string) []string {
				return []string{
					`data: {"type":"response.output_text.delta","delta":"hi"}`,
					`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"` + c5Field("usage", u) + `}}`,
				}
			},
			wire: func(un, ca int) string {
				return fmt.Sprintf(`{"input_tokens":%d,"output_tokens":0,"input_tokens_details":{"cached_tokens":%d}}`, un+ca, ca)
			},
		},
	}
}

func c5Field(name, value string) string {
	if value == "" {
		return ""
	}
	return `,"` + name + `":` + value
}

type c5Usage struct{ input, cacheRead int }

// c5Run returns the usage reported by Complete and by Stream's EventDone.
func c5Run(t *testing.T, a c5Adapter, usage string) (complete, stream c5Usage) {
	t.Helper()
	srv := jsonServer(a.complete(usage))
	defer srv.Close()
	client, err := a.newClient(srv.URL)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := client.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	complete = c5Usage{resp.InputTokens, resp.CacheReadTokens}

	ssrv := sseServer(a.stream(usage)...)
	defer ssrv.Close()
	sclient, err := a.newClient(ssrv.URL)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events, errs := collectStream(t, sclient)
	if len(errs) != 0 {
		t.Fatalf("stream errors: %v", errs)
	}
	for _, ev := range events {
		if ev.Type == llm.EventDone {
			return complete, c5Usage{ev.InputTokens, ev.CacheReadTokens}
		}
	}
	t.Fatal("no EventDone")
	return
}

// costOf prices a reported usage through the real accumulation path
// (Usage.AddResponse → EstimateCost) under the c5 test model.
func costOf(model string, u c5Usage) float64 {
	var usage llm.Usage
	usage.AddResponse(&llm.Response{Model: model, InputTokens: u.input, CacheReadTokens: u.cacheRead, OutputTokens: 0})
	return usage.Cost
}

// The same logical usage (1,000 uncached + 9,000 cached input tokens) must
// cost exactly the same under every adapter's wire shape, Complete and Stream.
// Pre-fix: Gemini charged 0.109 (cached billed twice), OpenAI-style 0.100
// (cached billed at the full input price); the right answer is 0.019.
func TestC5SameLogicalUsageSameCostAcrossProviders(t *testing.T) {
	model := c5Model(t)
	cases := []struct{ uncached, cached int }{
		{1_000, 9_000},
		{0, 50_000},  // fully cached prompt
		{12_345, 0},  // no cache hit
		{1, 999_999}, // nearly all cached
		{777_777, 1}, // single cached token
	}
	for _, a := range c5Adapters() {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/%d+%d", a.name, tc.uncached, tc.cached), func(t *testing.T) {
				want := c5WantCost(tc.uncached, tc.cached)
				comp, strm := c5Run(t, a, a.wire(tc.uncached, tc.cached))
				for path, u := range map[string]c5Usage{"Complete": comp, "Stream": strm} {
					if u.input != tc.uncached || u.cacheRead != tc.cached {
						t.Errorf("%s: InputTokens/CacheReadTokens = %d/%d, want %d/%d (InputTokens must exclude cached tokens)",
							path, u.input, u.cacheRead, tc.uncached, tc.cached)
					}
					if got := costOf(model, u); math.Abs(got-want) > 1e-12 {
						t.Errorf("%s: cost = %.9f, want %.9f (uncached × input + cached × cache-read, each token once)", path, got, want)
					}
				}
			})
		}
	}
}

// Hostile wire usage: a cached count larger than the prompt total, negative
// counts, and absent fields. InputTokens must never go negative, cached must
// never exceed the prompt the provider says it processed, and the cost must
// never exceed pricing the whole reported prompt at the full input rate.
func TestC5HostileCacheUsage(t *testing.T) {
	model := c5Model(t)
	type want struct{ input, cacheRead int }
	cases := map[string]map[string]struct {
		usage string
		want  want
	}{
		"gemini": {
			"cached>prompt":   {`{"promptTokenCount":100,"cachedContentTokenCount":500}`, want{0, 100}},
			"cached only":     {`{"cachedContentTokenCount":500}`, want{0, 0}},
			"negative cached": {`{"promptTokenCount":100,"cachedContentTokenCount":-50}`, want{100, 0}},
			"negative prompt": {`{"promptTokenCount":-100,"cachedContentTokenCount":50}`, want{0, 0}},
			"no cached field": {`{"promptTokenCount":100}`, want{100, 0}},
			"empty usage":     {`{}`, want{0, 0}},
			"MaxInt cached":   {fmt.Sprintf(`{"promptTokenCount":10,"cachedContentTokenCount":%d}`, math.MaxInt), want{0, 10}},
		},
		"openai_compat": {
			"cached>prompt":   {`{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":500}}`, want{0, 100}},
			"negative cached": {`{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":-50}}`, want{100, 0}},
			"negative prompt": {`{"prompt_tokens":-100,"prompt_tokens_details":{"cached_tokens":50}}`, want{0, 0}},
			"no details":      {`{"prompt_tokens":100}`, want{100, 0}},
			"null details":    {`{"prompt_tokens":100,"prompt_tokens_details":null}`, want{100, 0}},
			"empty details":   {`{"prompt_tokens":100,"prompt_tokens_details":{}}`, want{100, 0}},
			"MaxInt cached":   {fmt.Sprintf(`{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":%d}}`, math.MaxInt), want{0, 10}},
		},
		"openai_responses": {
			"cached>prompt":   {`{"input_tokens":100,"input_tokens_details":{"cached_tokens":500}}`, want{0, 100}},
			"negative cached": {`{"input_tokens":100,"input_tokens_details":{"cached_tokens":-50}}`, want{100, 0}},
			"negative prompt": {`{"input_tokens":-100,"input_tokens_details":{"cached_tokens":50}}`, want{0, 0}},
			"no details":      {`{"input_tokens":100}`, want{100, 0}},
			"null details":    {`{"input_tokens":100,"input_tokens_details":null}`, want{100, 0}},
			"MaxInt cached":   {fmt.Sprintf(`{"input_tokens":10,"input_tokens_details":{"cached_tokens":%d}}`, math.MaxInt), want{0, 10}},
		},
	}
	for _, a := range c5Adapters() {
		adapterCases, ok := cases[a.name]
		if !ok {
			continue
		}
		for name, tc := range adapterCases {
			t.Run(a.name+"/"+name, func(t *testing.T) {
				comp, strm := c5Run(t, a, tc.usage)
				for path, u := range map[string]c5Usage{"Complete": comp, "Stream": strm} {
					if u.input < 0 || u.cacheRead < 0 {
						t.Fatalf("%s: negative usage InputTokens=%d CacheReadTokens=%d", path, u.input, u.cacheRead)
					}
					if u.input != tc.want.input || u.cacheRead != tc.want.cacheRead {
						t.Errorf("%s: InputTokens/CacheReadTokens = %d/%d, want %d/%d", path, u.input, u.cacheRead, tc.want.input, tc.want.cacheRead)
					}
					ceiling := float64(tc.want.input+tc.want.cacheRead) * c5InputPrice / 1e6
					if got := costOf(model, u); got > ceiling+1e-12 {
						t.Errorf("%s: cost %.9f exceeds the whole prompt at the full input rate (%.9f)", path, got, ceiling)
					}
				}
			})
		}
	}
}

// Anthropic already reports the three input buckets disjointly; the adapter
// must pass them through untouched (no subtraction), including the
// "cache read larger than uncached input" shape that is perfectly normal
// there but would be "hostile" for the inclusive-total providers.
func TestC5AnthropicPassesDisjointBucketsThrough(t *testing.T) {
	model := c5Model(t)
	var anth c5Adapter
	for _, a := range c5Adapters() {
		if a.name == "anthropic" {
			anth = a
		}
	}
	usage := `{"input_tokens":100,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":50000}`
	comp, strm := c5Run(t, anth, usage)
	for path, u := range map[string]c5Usage{"Complete": comp, "Stream": strm} {
		if u.input != 100 || u.cacheRead != 50_000 {
			t.Errorf("%s: InputTokens/CacheReadTokens = %d/%d, want 100/50000", path, u.input, u.cacheRead)
		}
		if got, want := costOf(model, u), c5WantCost(100, 50_000); math.Abs(got-want) > 1e-12 {
			t.Errorf("%s: cost = %.9f, want %.9f", path, got, want)
		}
	}
	// Missing cache fields → 0.
	comp, strm = c5Run(t, anth, `{"input_tokens":100,"output_tokens":0}`)
	if comp != (c5Usage{100, 0}) || strm != (c5Usage{100, 0}) {
		t.Errorf("missing cache fields: Complete=%+v Stream=%+v, want {100 0}", comp, strm)
	}
}

// Moving cached tokens out of InputTokens must not make them FREE on a model
// with no registered cache-read price (most Gemini/OpenAI/DeepSeek entries):
// an unknown cache-read price bills cache reads at the input rate, so the
// split is cost-neutral there — the pre-C5 cost, never less.
func TestC5UnknownCacheReadPriceBillsAtInputRate(t *testing.T) {
	id := uniqueModelID("test-only-c5-nocacheprice")
	if err := llm.RegisterModel(llm.ModelInfo{ID: id, Provider: "custom", InputPricePer1M: c5InputPrice}); err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	got := llm.EstimateCost(llm.CostInput{Model: id, InputTokens: 1_000, CacheReadTokens: 9_000})
	want := 10_000 * c5InputPrice / 1e6
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("cost = %.9f, want %.9f — cached tokens on a model without a cache-read price must bill at the input rate, not $0", got, want)
	}
	// Hostile prices still never produce a negative/NaN cost.
	for _, p := range []float64{-1, math.NaN(), math.Inf(-1)} {
		bad := uniqueModelID("test-only-c5-badprice")
		if err := llm.RegisterModel(llm.ModelInfo{ID: bad, Provider: "custom", InputPricePer1M: p, CacheReadPricePer1M: p}); err != nil {
			t.Fatalf("RegisterModel: %v", err)
		}
		if c := llm.EstimateCost(llm.CostInput{Model: bad, InputTokens: 5, CacheReadTokens: 5}); c != 0 || math.IsNaN(c) {
			t.Errorf("price %v: cost = %v, want 0", p, c)
		}
	}
	// Built-in Gemini model without a cache price: a Gemini response with
	// cached tokens costs exactly what the whole prompt costs at input rate.
	info, ok := llm.GetModelInfo("gemini-2.5-pro")
	if !ok || info.CacheReadPricePer1M != 0 {
		t.Skip("gemini-2.5-pro now has a cache-read price; fallback case not exercised by this model")
	}
	got = llm.EstimateCost(llm.CostInput{Model: "gemini-2.5-pro", InputTokens: 1_000, CacheReadTokens: 9_000})
	if want := 10_000 * info.InputPricePer1M / 1e6; math.Abs(got-want) > 1e-12 {
		t.Fatalf("gemini-2.5-pro cost = %.9f, want %.9f", got, want)
	}
}
