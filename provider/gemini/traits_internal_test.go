package gemini

import (
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func buildFor(t *testing.T, model string, req llm.Request) (geminiRequest, error) {
	t.Helper()
	c := &Client{config: llm.Config{Model: model}, baseURL: defaultGeminiBase, providerName: "gemini"}
	if len(req.Messages) == 0 {
		req.Messages = []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}
	}
	return c.buildRequest(req)
}

// The thinking level sent per model: xhigh only on Hydrogen, never "minimal"
// (or its token budget) where the API rejects it.
func TestThinkingLevelPerModel(t *testing.T) {
	cases := []struct {
		model      string
		level      llm.ThinkingLevel
		wantLevel  string
		wantBudget bool
	}{
		{"gemini-4-hydrogen", llm.ThinkingXHigh, "xhigh", false},
		{"gemini-4-hydrogen", llm.ThinkingMax, "xhigh", false},
		{"gemini-4-hydrogen", llm.ThinkingHigh, "high", false},
		{"gemini-4-carbon", llm.ThinkingXHigh, "high", false},
		{"gemini-4-argon", llm.ThinkingMax, "high", false},
		{"gemini-3.8-flash", llm.ThinkingXHigh, "high", false},
		{"gemini-4-carbon", llm.ThinkingMinimal, "low", false},
		{"gemini-4-hydrogen", llm.ThinkingMinimal, "low", false},
		{"gemini-4-argon", llm.ThinkingMinimal, "low", false},
		{"gemini-3.8-flash", llm.ThinkingMinimal, "low", false},
		{"gemini-3.7-flash", llm.ThinkingMinimal, "low", false},
		{"flash", llm.ThinkingMinimal, "low", false}, // alias of gemini-3.8-flash
		{"gemini-4-neon", llm.ThinkingMinimal, "low", false},
		{"gemini-3.5-flash", llm.ThinkingMinimal, "", true}, // unchanged: a budget
	}
	for _, tc := range cases {
		got, err := buildFor(t, tc.model, llm.Request{ThinkingLevel: tc.level})
		if err != nil {
			t.Errorf("%s/%s: %v", tc.model, tc.level, err)
			continue
		}
		tcfg := got.GenerationConfig.ThinkingConfig
		if tcfg == nil {
			t.Errorf("%s/%s: no thinkingConfig", tc.model, tc.level)
			continue
		}
		if tcfg.ThinkingLevel != tc.wantLevel || (tcfg.ThinkingBudget > 0) != tc.wantBudget {
			t.Errorf("%s/%s: level %q budget %d, want level %q budget %v",
				tc.model, tc.level, tcfg.ThinkingLevel, tcfg.ThinkingBudget, tc.wantLevel, tc.wantBudget)
		}
	}
}

// An explicit ThinkingBudget still goes out as a budget on Gemini 4 (the API
// maps it to the nearest level with a warning).
func TestThinkingBudgetKeptOnGemini4(t *testing.T) {
	got, err := buildFor(t, "gemini-4-carbon", llm.Request{ThinkingLevel: llm.ThinkingMinimal, ThinkingBudget: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if tc := got.GenerationConfig.ThinkingConfig; tc.ThinkingBudget != 4096 || tc.ThinkingLevel != "" {
		t.Errorf("thinkingConfig %+v, want budget 4096 and no level", tc)
	}
}

// Explicit caching is refused before the request on Gemini 4 (the API
// answers it with an error), including an unregistered Gemini 4 ID, and
// still sent elsewhere.
func TestCachedContentRefusedOnGemini4(t *testing.T) {
	for _, model := range []string{"gemini-4-carbon", "gemini-4-hydrogen", "gemini-4-argon", "gemini-4-neon"} {
		_, err := buildFor(t, model, llm.Request{CachedContent: "cachedContents/abc"})
		if err == nil || !strings.Contains(err.Error(), "explicit caching") || !strings.Contains(err.Error(), model) {
			t.Errorf("%s: err = %v, want the explicit-caching error naming the model", model, err)
		}
		if _, err := buildFor(t, model, llm.Request{}); err != nil {
			t.Errorf("%s without CachedContent: %v", model, err)
		}
	}
	// Request.Model overrides the client's model in both directions.
	c := &Client{config: llm.Config{Model: "gemini-3.8-flash"}, baseURL: defaultGeminiBase, providerName: "gemini"}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}
	if _, err := c.buildRequest(llm.Request{Model: "gemini-4-carbon", Messages: msgs, CachedContent: "cachedContents/abc"}); err == nil {
		t.Error("Request.Model gemini-4-carbon with CachedContent was not refused")
	}
	got, err := buildFor(t, "gemini-3.8-flash", llm.Request{CachedContent: "cachedContents/abc"})
	if err != nil || got.CachedContent != "cachedContents/abc" {
		t.Errorf("gemini-3.8-flash: cachedContent %q, err %v; want it sent", got.CachedContent, err)
	}
}

// Thinking padding never lowers a long-decode model's maxOutputTokens to the
// per-request limit, and still caps the others at their output limit.
func TestMaxOutputTokensPadding(t *testing.T) {
	cases := []struct {
		model     string
		maxTokens int
		check     func(int) bool
		want      string
	}{
		{"gemini-4-carbon", 262144, func(n int) bool { return n >= 262144 }, ">= 262144"},
		{"gemini-4-hydrogen", 100000, func(n int) bool { return n >= 100000 }, ">= 100000"},
		{"gemini-4-carbon", 1000, func(n int) bool { return n > 1000 }, "> 1000 (padded)"},
		{"gemini-3.8-flash", 100000, func(n int) bool { return n == 65536 }, "65536"},
	}
	for _, tc := range cases {
		got, err := buildFor(t, tc.model, llm.Request{MaxTokens: tc.maxTokens})
		if err != nil {
			t.Fatal(err)
		}
		if n := got.GenerationConfig.MaxOutputTokens; !tc.check(n) {
			t.Errorf("%s MaxTokens %d: maxOutputTokens %d, want %s", tc.model, tc.maxTokens, n, tc.want)
		}
	}
}
