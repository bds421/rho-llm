package llm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
	"github.com/bds421/rho-llm/provider/gemini"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

// pauseServer answers the first request with a paused web search turn and
// every later one with the final turn, recording each request body.
type pauseServer struct {
	mu     sync.Mutex
	bodies [][]byte
	paused []byte
	final  []byte
	stream bool
}

func newPauseServer(t *testing.T) (*pauseServer, *httptest.Server) {
	t.Helper()
	ps := &pauseServer{
		paused: readFixture(t, "provider/anthropic/testdata/web_search_paused.json"),
		final:  readFixture(t, "provider/anthropic/testdata/web_search_final.json"),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		ps.mu.Lock()
		ps.bodies = append(ps.bodies, body)
		n := len(ps.bodies)
		stream := ps.stream
		ps.mu.Unlock()
		if stream && n == 1 {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write(readFixture(t, "provider/anthropic/testdata/web_search_stream.sse"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			_, _ = w.Write(ps.paused)
			return
		}
		_, _ = w.Write(ps.final)
	}))
	t.Cleanup(srv.Close)
	return ps, srv
}

func (ps *pauseServer) requests(t *testing.T) []map[string]json.RawMessage {
	t.Helper()
	ps.mu.Lock()
	defer ps.mu.Unlock()
	out := make([]map[string]json.RawMessage, len(ps.bodies))
	for i, b := range ps.bodies {
		if err := json.Unmarshal(b, &out[i]); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

func readFixture(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func pauseClient(t *testing.T, base string) llm.Client {
	t.Helper()
	c, err := anthropic.New(llm.Config{Provider: "anthropic", Model: "claude-opus-5-5", APIKey: "test-key", BaseURL: base, Timeout: 10 * time.Second, MaxTokens: 4096})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

var searchTools = []llm.ServerTool{{
	Type: llm.WebSearch20260209, Name: llm.ServerToolWebSearch, MaxUses: 4,
	AllowedDomains: []string{"example.com", "example.org"},
	UserLocation:   &llm.UserLocation{Country: "US", Timezone: "America/New_York"},
	AllowedCallers: []string{llm.CallerDirect},
	CacheControl:   true,
}}

// checkContinuation asserts the continuation request is the first request
// with the paused assistant content appended: tools and system byte for
// byte, history unchanged, no extra user message, the container reused.
func checkContinuation(t *testing.T, reqs []map[string]json.RawMessage) {
	t.Helper()
	if len(reqs) != 2 {
		t.Fatalf("requests = %d, want 2", len(reqs))
	}
	first, second := reqs[0], reqs[1]
	for _, key := range []string{"tools", "system", "model", "max_tokens"} {
		if !bytes.Equal(first[key], second[key]) {
			t.Errorf("%s changed across the continuation:\n%s\n%s", key, first[key], second[key])
		}
	}
	if len(first["tools"]) == 0 {
		t.Fatal("no tools sent")
	}
	if string(second["container"]) != `"container_01"` {
		t.Errorf("container = %s", second["container"])
	}
	var m1, m2 []map[string]json.RawMessage
	_ = json.Unmarshal(first["messages"], &m1)
	_ = json.Unmarshal(second["messages"], &m2)
	if len(m2) != len(m1)+1 {
		t.Fatalf("continuation has %d messages, want %d", len(m2), len(m1)+1)
	}
	for i := range m1 {
		if !bytes.Equal(m1[i]["content"], m2[i]["content"]) {
			t.Errorf("message %d changed", i)
		}
	}
	last := m2[len(m2)-1]
	if string(last["role"]) != `"assistant"` {
		t.Fatalf("last role = %s (a continuation must not add a user message)", last["role"])
	}
	var want struct {
		Content any `json:"content"`
	}
	_ = json.Unmarshal(readFixture(t, "provider/anthropic/testdata/web_search_paused.json"), &want)
	var got any
	_ = json.Unmarshal(last["content"], &got)
	if !reflect.DeepEqual(got, want.Content) {
		t.Fatalf("paused content not sent back unchanged:\n got %v\nwant %v", got, want.Content)
	}
}

func TestSessionContinuePausedTurn(t *testing.T) {
	ps, srv := newPauseServer(t)
	s := llm.NewSession(pauseClient(t, srv.URL),
		llm.WithSystem("Answer with sources."),
		llm.WithServerTools(searchTools...),
		llm.WithBaseRequest(llm.Request{SystemCacheControl: true}),
	)
	resp, err := s.Send(context.Background(), "When did the example widget ship?")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != llm.StopPauseTurn {
		t.Fatalf("StopReason = %q", resp.StopReason)
	}
	final, err := s.Continue(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if final.StopReason != llm.StopEndTurn {
		t.Fatalf("final StopReason = %q", final.StopReason)
	}
	checkContinuation(t, ps.requests(t))

	if _, err := s.Continue(context.Background()); !errors.Is(err, llm.ErrNotPaused) {
		t.Fatalf("Continue after end_turn: err = %v, want ErrNotPaused", err)
	}
	if n := len(ps.requests(t)); n != 2 {
		t.Fatalf("a refused Continue sent a request (%d requests)", n)
	}

	u := s.Usage()
	if u.ServerToolUse.WebSearchRequests != 1 {
		t.Errorf("WebSearchRequests = %d", u.ServerToolUse.WebSearchRequests)
	}
	// Paused turn: Opus 5.5 tokens + 1 search. Final turn: priced per
	// iteration (Opus 5.5 attempt + Opus 4.8 fallback), not at its top level.
	want := (6039*4.00+931*20.00+1200*5.00+7123*0.20)/1e6 + 0.01 +
		(535*4.00)/1e6 + (412*5.00+264*25.00)/1e6
	if math.Abs(u.Cost-want) > 1e-12 {
		t.Errorf("Cost = %.10f, want %.10f", u.Cost, want)
	}
}

func TestSessionStreamContinuePausedTurn(t *testing.T) {
	ps, srv := newPauseServer(t)
	ps.stream = true
	s := llm.NewSession(pauseClient(t, srv.URL), llm.WithSystem("Answer with sources."), llm.WithServerTools(searchTools...))
	for _, err := range s.Stream(context.Background(), "When did the example widget ship?") {
		if err != nil {
			t.Fatal(err)
		}
	}
	conv := s.Conversation()
	if last := conv.Messages[len(conv.Messages)-1]; last.StopReason != llm.StopPauseTurn || last.ContainerID != "container_01" {
		t.Fatalf("last message = %+v", last)
	}
	final, err := s.Continue(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if final.StopReason != llm.StopEndTurn {
		t.Fatalf("final = %q", final.StopReason)
	}
	checkContinuation(t, ps.requests(t))
}

func TestStreamContinueRefusesUnpausedTurn(t *testing.T) {
	ps, srv := newPauseServer(t)
	s := llm.NewSession(pauseClient(t, srv.URL))
	var gotErr error
	for _, err := range s.StreamContinue(context.Background()) {
		gotErr = err
	}
	if !errors.Is(gotErr, llm.ErrNotPaused) {
		t.Fatalf("err = %v", gotErr)
	}
	if len(ps.requests(t)) != 0 {
		t.Fatal("request sent for an empty conversation")
	}
}

func TestContinuePausedTurnKeepsToolsByteIdentical(t *testing.T) {
	ps, srv := newPauseServer(t)
	c := pauseClient(t, srv.URL)
	req := llm.Request{
		System:             "Answer with sources.",
		SystemCacheControl: true,
		Messages:           []llm.Message{llm.NewTextMessage(llm.RoleUser, "When did the example widget ship?")},
		Tools:              []llm.Tool{{Name: "lookup", Description: "Look up a record.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"type": "string"}}}, CacheControl: true}},
		ServerTools:        searchTools,
	}
	resp, err := c.Complete(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	next, err := llm.ContinuePausedTurn(req, resp)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 1 {
		t.Fatal("ContinuePausedTurn modified the caller's request")
	}
	if _, err := c.Complete(context.Background(), next); err != nil {
		t.Fatal(err)
	}
	checkContinuation(t, ps.requests(t))

	final := &llm.Response{StopReason: llm.StopEndTurn}
	if _, err := llm.ContinuePausedTurn(next, final); !errors.Is(err, llm.ErrNotPaused) {
		t.Fatalf("err = %v", err)
	}
	if _, err := llm.ContinuePausedTurn(next, nil); err == nil {
		t.Fatal("nil response accepted")
	}
}

// TestSessionContinueConcurrent races Continue against Send and snapshot
// reads; run under -race. Every Continue either succeeds on a paused turn or
// returns ErrNotPaused — never sends a trailing assistant message otherwise.
func TestSessionContinueConcurrent(t *testing.T) {
	_, srv := newPauseServer(t)
	s := llm.NewSession(pauseClient(t, srv.URL), llm.WithServerTools(searchTools...))
	if _, err := s.Send(context.Background(), "q"); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	continued := 0
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			_, err := s.Continue(context.Background())
			if err == nil {
				mu.Lock()
				continued++
				mu.Unlock()
			} else if !errors.Is(err, llm.ErrNotPaused) {
				t.Error(err)
			}
		}()
		go func() {
			defer wg.Done()
			_ = s.Conversation()
			_ = s.Usage()
		}()
	}
	wg.Wait()
	if continued != 1 {
		t.Fatalf("continued %d times, want exactly 1", continued)
	}
}

func TestEstimateCostServerToolsAndIterations(t *testing.T) {
	base := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", InputTokens: 1000, OutputTokens: 100})
	withSearch := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", InputTokens: 1000, OutputTokens: 100, WebSearchRequests: 3})
	if math.Abs(withSearch-base-0.03) > 1e-12 {
		t.Errorf("3 searches added %.6f, want 0.03", withSearch-base)
	}
	batch := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", InputTokens: 1000, OutputTokens: 100, WebSearchRequests: 3, Batch: true})
	if math.Abs(batch-(base/2+0.03)) > 1e-12 {
		t.Errorf("batch = %.6f: searches must not get the batch discount", batch)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "unknown-model", WebSearchRequests: 10}); got != 0 {
		t.Errorf("unknown model priced searches: %v", got)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "gemini-2.5-flash", WebSearchRequests: 10}); got != 0 {
		t.Errorf("model without a search price priced searches: %v", got)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", WebSearchRequests: -5}); got != 0 {
		t.Errorf("negative searches = %v", got)
	}

	iter := llm.EstimateCost(llm.CostInput{
		Model: "claude-opus-5-5", InputTokens: 999999, OutputTokens: 999999, // ignored when iterations are set
		Iterations: []llm.UsageIteration{
			{Type: "compaction", InputTokens: 1000, CacheReadTokens: 2000},
			{Type: "fallback_message", Model: "claude-opus-4-8", InputTokens: 100, OutputTokens: 10, CacheCreationTokens: 50},
			{Type: "message", Model: "no-such-model", InputTokens: 1 << 30},
			{Type: "message", InputTokens: -7, OutputTokens: llm.TokensNotReported},
		},
	})
	want := (1000*4.00+2000*0.20)/1e6 + (100*5.00+10*25.00+50*6.25)/1e6
	if math.Abs(iter-want) > 1e-12 {
		t.Errorf("iterations = %.10f, want %.10f", iter, want)
	}
	huge := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", WebSearchRequests: math.MaxInt, Iterations: []llm.UsageIteration{
		{InputTokens: math.MaxInt, OutputTokens: math.MaxInt, CacheReadTokens: math.MaxInt, CacheCreationTokens: math.MaxInt},
		{InputTokens: math.MaxInt, OutputTokens: math.MaxInt, CacheReadTokens: math.MaxInt, CacheCreationTokens: math.MaxInt},
	}})
	if math.IsInf(huge, 0) || math.IsNaN(huge) || huge <= 0 {
		t.Errorf("saturated cost = %v", huge)
	}
}

func TestNormalizeKeepsServerToolsWithTheirProvider(t *testing.T) {
	assistant := llm.Message{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
		{Type: llm.ContentServerToolUse, ToolUseID: "srvtoolu_1", ToolName: "web_search", ProviderBlock: json.RawMessage(`{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}`)},
		{Type: llm.ContentServerToolResult, ToolResultID: "srvtoolu_1", BlockType: "web_search_tool_result", ProviderBlock: json.RawMessage(`{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]}`)},
		{Type: llm.ContentProviderBlock, BlockType: "fallback", ProviderBlock: json.RawMessage(`{"type":"fallback"}`)},
		{Type: llm.ContentText, Text: "answer", Citations: []llm.Citation{{Type: llm.CitationWebSearchResultLocation, URL: "https://example.com"}}},
	}}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "q"), assistant}

	same := llm.NormalizeForProvider(msgs, "anthropic")
	if len(same[1].Content) != 4 || len(same[1].Content[3].Citations) != 1 {
		t.Fatalf("same provider lost content: %+v", same[1].Content)
	}
	other := llm.NormalizeForProvider(msgs, "gemini")
	if len(other[1].Content) != 1 || other[1].Content[0].Text != "answer" || other[1].Content[0].Citations != nil {
		t.Fatalf("handoff content = %+v", other[1].Content)
	}
	if len(assistant.Content[3].Citations) != 1 {
		t.Fatal("normalization mutated the input")
	}
}

func TestOtherAdaptersRejectServerTools(t *testing.T) {
	cfg := func(provider, model string) llm.Config {
		return llm.Config{Provider: provider, Model: model, APIKey: "k", BaseURL: "http://127.0.0.1:1", Timeout: time.Second}
	}
	clients := map[string]func() (llm.Client, error){
		"gemini":           func() (llm.Client, error) { return gemini.New(cfg("gemini", "gemini-2.5-flash")) },
		"openai_compat":    func() (llm.Client, error) { return openaicompat.New(cfg("openai", "gpt-4o")) },
		"openai_responses": func() (llm.Client, error) { return openairesponses.New(cfg("openai", "gpt-5")) },
	}
	for name, newClient := range clients {
		t.Run(name, func(t *testing.T) {
			c, err := newClient()
			if err != nil {
				t.Fatal(err)
			}
			req := llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "q")}, ServerTools: searchTools}
			if _, err := c.Complete(context.Background(), req); err == nil || !bytes.Contains([]byte(err.Error()), []byte("server tools")) {
				t.Fatalf("Complete err = %v", err)
			}
			var streamErr error
			for _, err := range c.Stream(context.Background(), req) {
				streamErr = err
			}
			if streamErr == nil {
				t.Fatal("Stream accepted server tools")
			}
		})
	}
}

func TestServerToolsRequireToolCapability(t *testing.T) {
	caps := llm.RequestCapabilities(llm.Request{ServerTools: searchTools}, false)
	for _, c := range caps {
		if c == llm.CapabilityTools {
			return
		}
	}
	t.Fatalf("capabilities = %v, want CapabilityTools for a server tool", caps)
}

func TestEstimateCostOneHourCacheWrites(t *testing.T) {
	five := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", CacheCreateTokens: 1_000_000})
	mixed := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", CacheCreateTokens: 1_000_000, CacheCreate1hTokens: 250_000})
	if math.Abs(five-5.00) > 1e-9 || math.Abs(mixed-(0.75*5.00+0.25*8.00)) > 1e-9 {
		t.Fatalf("5m-only = %v, mixed = %v", five, mixed)
	}
	capped := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", CacheCreateTokens: 100, CacheCreate1hTokens: 1 << 40})
	if math.Abs(capped-100*8.00/1e6) > 1e-12 {
		t.Fatalf("1h share above the total = %v, want it capped at the total", capped)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "claude-opus-5-5", CacheCreateTokens: 100, CacheCreate1hTokens: -3}); math.Abs(got-100*5.00/1e6) > 1e-12 {
		t.Fatalf("negative 1h share = %v", got)
	}

	if err := llm.RegisterModel(llm.ModelInfo{ID: "test-no-1h-price", Provider: "anthropic", InputPricePer1M: 1, CacheWritePricePer1M: 1.25}); err != nil {
		t.Fatal(err)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "test-no-1h-price", CacheCreateTokens: 1_000_000, CacheCreate1hTokens: 1_000_000}); math.Abs(got-2.00) > 1e-9 {
		t.Fatalf("unknown 1h price: cost = %v, want 2x input", got)
	}
	if err := llm.RegisterModel(llm.ModelInfo{ID: "test-no-prices", Provider: "anthropic", CacheWritePricePer1M: 1.25}); err != nil {
		t.Fatal(err)
	}
	if got := llm.EstimateCost(llm.CostInput{Model: "test-no-prices", CacheCreateTokens: 1_000_000, CacheCreate1hTokens: 1_000_000}); got != 0 {
		t.Fatalf("no input price: 1h writes cost %v, want 0", got)
	}
}

func TestSessionStreamCarriesOneHourWrites(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_creation_input_tokens\":300,\"cache_creation\":{\"ephemeral_5m_input_tokens\":100,\"ephemeral_1h_input_tokens\":200}}}}\n\n"+
			"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n")
	}))
	defer srv.Close()
	s := llm.NewSession(pauseClient(t, srv.URL))
	for _, err := range s.Stream(context.Background(), "q") {
		if err != nil {
			t.Fatal(err)
		}
	}
	u := s.Usage()
	want := (10*4.0 + 5*20.0 + 100*5.0 + 200*8.0) / 1e6
	if u.CacheCreation1hTokens != 200 || math.Abs(u.Cost-want) > 1e-12 {
		t.Fatalf("usage = %+v, want cost %.10f", u, want)
	}
}
