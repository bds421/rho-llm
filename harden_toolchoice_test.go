package llm_test

// Hardening pass 16 — adversarial coverage for ToolChoice, SamplingParams and
// RawStopReason (the pi `ai` gap-closing pass). NO happy-path assertions for
// their own sake: every test encodes a way the new wiring can silently break —
// a constraint that never reaches the wire, a passthrough key that hijacks the
// request, a normalized stop reason that erases what the provider actually
// said — and asserts the break is absent. Each was proven to go red against an
// injected regression before being committed.

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	llm "github.com/bds421/rho-llm"
	"github.com/bds421/rho-llm/provider/anthropic"
	"github.com/bds421/rho-llm/provider/gemini"
	"github.com/bds421/rho-llm/provider/openaicompat"
	"github.com/bds421/rho-llm/provider/openairesponses"
)

// tcProtocols pairs each wire protocol with a constructor, a model that routes
// to it, and the minimal valid response body that protocol returns.
var tcProtocols = []struct {
	name  string
	model string
	resp  string
	new   func(llm.Config) (llm.Client, error)
}{
	{
		name: "openai_compat", model: "gpt-4.1",
		resp: `{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		new:  func(c llm.Config) (llm.Client, error) { return openaicompat.New(c) },
	},
	{
		name: "anthropic", model: "claude-opus-5",
		resp: `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"m","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
		new:  func(c llm.Config) (llm.Client, error) { return anthropic.New(c) },
	},
	{
		name: "gemini", model: "gemini-2.5-flash",
		resp: `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`,
		new:  func(c llm.Config) (llm.Client, error) { return gemini.New(c) },
	},
	{
		name: "openai_responses", model: "gpt-5.6-sol",
		resp: `{"id":"resp_1","object":"response","model":"m","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`,
		new:  func(c llm.Config) (llm.Client, error) { return openairesponses.New(c) },
	},
}

// tcDo issues one Complete against a capturing server and returns the decoded
// request body. It fails the test if the request errored (unless wantErr).
func tcDo(t *testing.T, proto int, req llm.Request, wantErr bool) map[string]any {
	t.Helper()
	p := tcProtocols[proto]
	srv, raw := captureBody(t, p.resp)
	defer srv.Close()
	c, err := p.new(llm.Config{Provider: p.name, Model: p.model, APIKey: "k", BaseURL: srv.URL, Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("%s New: %v", p.name, err)
	}
	defer c.Close()
	req.MaxTokens = 16
	req.Messages = []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}
	_, err = c.Complete(context.Background(), req)
	if wantErr {
		if err == nil {
			t.Fatalf("%s: expected an error, got none", p.name)
		}
		return nil
	}
	if err != nil {
		t.Fatalf("%s Complete: %v", p.name, err)
	}
	if len(*raw) == 0 {
		t.Fatalf("%s: server never received a body — assertion would be vacuous", p.name)
	}
	var m map[string]any
	if err := json.Unmarshal(*raw, &m); err != nil {
		t.Fatalf("%s: body is not a JSON object: %v (%s)", p.name, err, *raw)
	}
	return m
}

// tcField is the wire field each protocol carries tool choice in.
func tcField(proto int) string {
	if tcProtocols[proto].name == "gemini" {
		return "toolConfig"
	}
	return "tool_choice"
}

func oneTool() []llm.Tool {
	return []llm.Tool{{
		Name:        "get_weather",
		Description: "weather",
		InputSchema: map[string]any{"type": "object"},
	}}
}

// A ToolChoice that constrains tool use MUST reach the wire. The bug this
// guards: a caller says "you must call a tool", the adapter drops it, and the
// model answers with prose instead — a silent, provider-side behavior change
// that no error ever surfaces.
func TestToolChoiceReachesTheWire(t *testing.T) {
	for i, p := range tcProtocols {
		for _, mode := range []llm.ToolChoiceMode{llm.ToolChoiceNone, llm.ToolChoiceRequired, llm.ToolChoiceTool} {
			t.Run(p.name+"/"+string(mode), func(t *testing.T) {
				tc := &llm.ToolChoice{Mode: mode}
				if mode == llm.ToolChoiceTool {
					tc.Name = "get_weather"
				}
				got := tcDo(t, i, llm.Request{Tools: oneTool(), ToolChoice: tc}, false)
				v, ok := got[tcField(i)]
				if !ok {
					t.Fatalf("%s: ToolChoice{%s} never reached the wire — no %q field in %v",
						p.name, mode, tcField(i), got)
				}
				if mode == llm.ToolChoiceTool {
					if enc, _ := json.Marshal(v); !strings.Contains(string(enc), "get_weather") {
						t.Errorf("%s: forced tool name lost on the wire: %s", p.name, enc)
					}
				}
			})
		}
	}
}

// Auto is every provider's default, so it must NOT be emitted. The bug this
// guards: shipping an explicit "auto" to a provider that rejects the field, or
// that treats an explicit value differently from its absence.
func TestToolChoiceAutoIsOmitted(t *testing.T) {
	for i, p := range tcProtocols {
		t.Run(p.name, func(t *testing.T) {
			got := tcDo(t, i, llm.Request{Tools: oneTool(), ToolChoice: llm.NewToolChoice(llm.ToolChoiceAuto)}, false)
			for _, field := range []string{"tool_choice", "toolConfig"} {
				if _, present := got[field]; present {
					t.Errorf("%s: ToolChoiceAuto emitted %q — auto is the default and must be omitted", p.name, field)
				}
			}
		})
	}
}

// A tool constraint with NO tools is a 400 on real providers, so the adapter
// must not ship one. The bug this guards: a caller sets ToolChoice but forgets
// Tools, and every request fails at the provider with an opaque error.
func TestToolChoiceWithoutToolsIsNotSent(t *testing.T) {
	for i, p := range tcProtocols {
		t.Run(p.name, func(t *testing.T) {
			got := tcDo(t, i, llm.Request{ToolChoice: llm.NewToolChoice(llm.ToolChoiceRequired)}, false)
			for _, field := range []string{"tool_choice", "toolConfig"} {
				if _, present := got[field]; present {
					t.Errorf("%s: sent %q with an empty tools array — providers reject this", p.name, field)
				}
			}
		})
	}
}

// A malformed ToolChoice must fail LOUDLY at the library boundary, not sail
// through as an opaque provider 400 — or, worse, as a silently ignored field.
func TestMalformedToolChoiceIsRejected(t *testing.T) {
	bad := []struct {
		name string
		tc   *llm.ToolChoice
	}{
		{"mode tool with no name", &llm.ToolChoice{Mode: llm.ToolChoiceTool}},
		{"empty mode", &llm.ToolChoice{}},
		{"unknown mode", &llm.ToolChoice{Mode: "definitely-not-a-mode"}},
		{"name without tool mode", &llm.ToolChoice{Mode: llm.ToolChoiceRequired, Name: "get_weather"}},
		{"name on auto", &llm.ToolChoice{Mode: llm.ToolChoiceAuto, Name: "get_weather"}},
	}
	for i, p := range tcProtocols {
		for _, b := range bad {
			t.Run(p.name+"/"+b.name, func(t *testing.T) {
				tcDo(t, i, llm.Request{Tools: oneTool(), ToolChoice: b.tc}, true)
			})
		}
	}
}

// A nil ToolChoice is the documented "provider default" and must never panic or
// error — it is the overwhelmingly common path.
func TestNilToolChoiceIsInert(t *testing.T) {
	if err := (*llm.ToolChoice)(nil).Validate(); err != nil {
		t.Fatalf("nil ToolChoice must validate clean, got %v", err)
	}
	for i, p := range tcProtocols {
		t.Run(p.name, func(t *testing.T) {
			got := tcDo(t, i, llm.Request{Tools: oneTool()}, false)
			for _, field := range []string{"tool_choice", "toolConfig"} {
				if _, present := got[field]; present {
					t.Errorf("%s: nil ToolChoice still emitted %q", p.name, field)
				}
			}
		})
	}
}
