package openaicompat

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func body(t *testing.T, c *Client, req llm.Request) map[string]any {
	t.Helper()
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	raw, _ := json.Marshal(apiReq)
	var out map[string]any
	_ = json.Unmarshal(raw, &out)
	return out
}

func hi() []llm.Message { return []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")} }

// ThinkingLevel was refused on every OpenAI-compatible host. Each host now
// gets its own documented field (P2), and only that field.
func TestThinkingLevelPerHostDialect(t *testing.T) {
	reasoningKeys := []string{"reasoning_effort", "thinking", "enable_thinking", "reasoning"}
	cases := []struct {
		provider, baseURL, model string
		want                     map[string]any
	}{
		{"deepseek", "", "deepseek-flash", map[string]any{"thinking": map[string]any{"type": "enabled"}, "reasoning_effort": "high"}},
		{"kimi", "", "kimi-k3.1-pro", map[string]any{"reasoning_effort": "high"}},
		{"moonshot", "", "kimi-k2.6", map[string]any{"thinking": map[string]any{"type": "enabled"}}},
		{"glm", "", "glm-5.3", map[string]any{"thinking": map[string]any{"type": "enabled"}}},
		{"qwen-cn", "", "qwen3.6-plus", map[string]any{"enable_thinking": true}},
		{"openrouter", "", "deepseek/deepseek-v4", map[string]any{"reasoning": map[string]any{"effort": "high"}}},
		{"together", "", "Qwen/Qwen3.8", map[string]any{"reasoning": map[string]any{"enabled": true}}},
		{"groq", "", "openai/gpt-oss-120b", map[string]any{"reasoning_effort": "high"}},
		// A custom name is recognised by its BaseURL.
		{"my-gateway", "https://api.deepseek.com/v1", "deepseek-flash", map[string]any{"thinking": map[string]any{"type": "enabled"}, "reasoning_effort": "high"}},
	}
	for _, tc := range cases {
		t.Run(tc.provider, func(t *testing.T) {
			c := &Client{config: llm.Config{Model: tc.model}, providerName: tc.provider, baseURL: tc.baseURL}
			got := body(t, c, llm.Request{MaxTokens: 10, ThinkingLevel: llm.ThinkingHigh, Messages: hi()})
			for _, k := range reasoningKeys {
				wantV, want := tc.want[k]
				gotV, has := got[k]
				if want != has {
					t.Errorf("%s: present=%v, want %v (body %v)", k, has, want, got)
					continue
				}
				if want {
					wb, _ := json.Marshal(wantV)
					gb, _ := json.Marshal(gotV)
					if string(wb) != string(gb) {
						t.Errorf("%s = %s, want %s", k, gb, wb)
					}
				}
			}
			// No level: nothing is sent, the host default applies.
			none := body(t, c, llm.Request{MaxTokens: 10, Messages: hi()})
			for _, k := range reasoningKeys {
				if _, has := none[k]; has {
					t.Errorf("no level: %s sent", k)
				}
			}
		})
	}
	// A host without a documented control refuses the level.
	for _, p := range []string{"mistral", "cohere", "xai", "minimax", "unknown-host"} {
		c := &Client{config: llm.Config{Model: "m"}, providerName: p, baseURL: "https://example.invalid/v1"}
		if _, err := c.buildRequest(llm.Request{MaxTokens: 10, ThinkingLevel: llm.ThinkingLow, Messages: hi()}, false); err == nil {
			t.Errorf("%s: ThinkingLevel accepted", p)
		}
	}
	// A level and the same field through SamplingParams is a conflict, not a
	// silent overwrite.
	c := &Client{config: llm.Config{Model: "deepseek-flash"}, providerName: "deepseek"}
	apiReq, err := c.buildRequest(llm.Request{MaxTokens: 10, ThinkingLevel: llm.ThinkingHigh, Messages: hi()}, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := llm.MergeSamplingParams(apiReq, map[string]any{"reasoning_effort": "low"}); err == nil {
		t.Error("SamplingParams reasoning_effort silently overrode ThinkingLevel")
	}
}

func toolHistory(provider, field, reasoning string) []llm.Message {
	return []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "q"),
		{Role: llm.RoleAssistant, Provider: provider, Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: reasoning, ReasoningField: field},
			{Type: llm.ContentToolUse, ToolUseID: "call_1", ToolName: "f", ToolInput: map[string]any{}},
		}},
		llm.NewToolResultMessage("call_1", "ok", false),
	}
}

func assistant(t *testing.T, b map[string]any) map[string]any {
	t.Helper()
	for _, m := range b["messages"].([]any) {
		if mm := m.(map[string]any); mm["role"] == "assistant" {
			return mm
		}
	}
	t.Fatalf("no assistant message in %v", b)
	return nil
}

// DeepSeek answers a 400 when a tool conversation does not pass
// reasoning_content back; the adapter dropped it (P2). Hosts that do not
// document taking it back (Groq) must not get it: they reject unknown fields.
func TestReasoningReplayedOnlyWhereTaken(t *testing.T) {
	ds := &Client{config: llm.Config{Model: "deepseek-flash"}, providerName: "deepseek"}
	a := assistant(t, body(t, ds, llm.Request{MaxTokens: 10, Messages: toolHistory("deepseek", "reasoning_content", "think")}))
	if a["reasoning_content"] != "think" {
		t.Errorf("deepseek: reasoning_content = %v", a["reasoning_content"])
	}
	// A DeepSeek assistant turn without reasoning still carries the field.
	a = assistant(t, body(t, ds, llm.Request{MaxTokens: 10, Messages: toolHistory("deepseek", "", "")}))
	if v, has := a["reasoning_content"]; !has || v != "" {
		t.Errorf("deepseek without reasoning: reasoning_content = %v (present %v), want \"\"", v, has)
	}
	or := &Client{config: llm.Config{Model: "m"}, providerName: "openrouter"}
	a = assistant(t, body(t, or, llm.Request{MaxTokens: 10, Messages: toolHistory("openrouter", "reasoning", "think")}))
	if a["reasoning"] != "think" || a["reasoning_content"] != nil {
		t.Errorf("openrouter: reasoning=%v reasoning_content=%v, want it under the field it came in", a["reasoning"], a["reasoning_content"])
	}
	groq := &Client{config: llm.Config{Model: "m"}, providerName: "groq"}
	a = assistant(t, body(t, groq, llm.Request{MaxTokens: 10, Messages: toolHistory("groq", "reasoning", "think")}))
	if a["reasoning"] != nil || a["reasoning_content"] != nil {
		t.Errorf("groq got reasoning back: %v", a)
	}
	// Only the two known field names are ever written, and never empty text.
	for _, field := range []string{"content", "role", "tool_calls"} {
		a = assistant(t, body(t, ds, llm.Request{MaxTokens: 10, Messages: toolHistory("deepseek", field, "x")}))
		if a["reasoning_content"] != "" || a[field] == "x" {
			t.Errorf("ReasoningField %q: assistant = %v", field, a)
		}
	}
}

// End to end through a Session: the reasoning DeepSeek returns on a tool turn
// (streamed and not) is in the next request, and a provider switch turns it
// into text instead.
func TestDeepSeekSessionCarriesReasoningIntoNextTurn(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]any
	turn := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var b map[string]any
		_ = json.Unmarshal(raw, &b)
		mu.Lock()
		bodies = append(bodies, b)
		turn++
		n := turn
		mu.Unlock()
		stream := b["stream"] == true
		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
		}
		switch {
		case n%2 == 1 && stream:
			_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"reasoning_content":"need "}}]}`+"\n\n"+
				`data: {"choices":[{"index":0,"delta":{"reasoning_content":"the tool"}}]}`+"\n\n"+
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\ndata: [DONE]\n\n")
		case n%2 == 1:
			_, _ = io.WriteString(w, `{"id":"a","model":"deepseek-flash","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":"need the tool","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
		case stream:
			_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\ndata: [DONE]\n\n")
		default:
			_, _ = io.WriteString(w, `{"id":"b","model":"deepseek-flash","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
		}
	}))
	defer srv.Close()
	tools := []llm.Tool{{Name: "f", InputSchema: map[string]any{"type": "object"}}}
	for _, streamed := range []bool{false, true} {
		c, err := New(llm.Config{Provider: "deepseek", Model: "deepseek-flash", APIKey: "k", BaseURL: srv.URL})
		if err != nil {
			t.Fatal(err)
		}
		s := llm.NewSession(c, llm.WithBaseRequest(llm.Request{Tools: tools, MaxTokens: 100, ThinkingLevel: llm.ThinkingHigh}))
		send := func(m llm.Message) {
			t.Helper()
			if !streamed {
				if _, err := s.SendMessages(t.Context(), m); err != nil {
					t.Fatal(err)
				}
				return
			}
			for _, err := range s.StreamMessages(t.Context(), m) {
				if err != nil {
					t.Fatal(err)
				}
			}
		}
		send(llm.NewTextMessage(llm.RoleUser, "q"))
		send(llm.NewToolResultMessage("call_1", "ok", false))
		mu.Lock()
		last := bodies[len(bodies)-1]
		mu.Unlock()
		if a := assistant(t, last); a["reasoning_content"] != "need the tool" {
			t.Errorf("streamed=%v: second request assistant = %v", streamed, a)
		}
		// The transcript carries the field; a handoff degrades it to text.
		msgs := s.Conversation().Messages
		other := llm.NormalizeForProvider(msgs, "groq")
		raw, _ := json.Marshal(other)
		if strings.Contains(string(raw), "reasoning_field") || !strings.Contains(string(raw), "need the tool") {
			t.Errorf("streamed=%v: handoff transcript = %s", streamed, raw)
		}
	}
}
