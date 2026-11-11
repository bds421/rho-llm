package llm_test

import (
	"encoding/json"
	"reflect"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestNormalizeNilAndEmpty(t *testing.T) {
	if got := llm.NormalizeForProvider(nil, ""); len(got) != 0 {
		t.Errorf("nil input -> %v", got)
	}
	if got := llm.NormalizeForProvider([]llm.Message{}, "anthropic"); len(got) != 0 {
		t.Errorf("empty input -> %v", got)
	}
}

func assistantTool(provider, id string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Provider: provider, StopReason: "tool_use", Content: []llm.ContentPart{
		{Type: llm.ContentToolUse, ToolUseID: id, ToolName: "f", ToolInput: map[string]any{}},
	}}
}

func TestNormalizeDropsDanglingToolResult(t *testing.T) {
	// A tool result whose id matches NO tool call must be dropped.
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "hi"),
		llm.NewToolResultMessage("ghost", "stale", false),
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolResult {
				t.Errorf("dangling tool result survived: %+v", p)
			}
		}
	}
}

func TestNormalizeMultipleOrphanCalls(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", StopReason: "tool_use", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "a", ToolName: "f"},
			{Type: llm.ContentToolUse, ToolUseID: "b", ToolName: "g"},
		}},
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")
	got := map[string]bool{}
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolResult && p.IsError {
				got[p.ToolResultID] = true
			}
		}
	}
	if !got["a"] || !got["b"] {
		t.Errorf("expected synthetic results for both orphan calls, got %v", got)
	}
}

func TestNormalizeErroredTurnDropsItsResult(t *testing.T) {
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "q"),
		{Role: llm.RoleAssistant, Provider: "anthropic", StopReason: "error", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "dead", ToolName: "f"},
		}},
		llm.NewToolResultMessage("dead", "result-for-dead-call", false),
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")
	for _, m := range out {
		if m.Role == llm.RoleAssistant {
			t.Errorf("errored turn survived: %+v", m)
		}
		for _, p := range m.Content {
			if p.Type == llm.ContentToolResult && p.ToolResultID == "dead" {
				t.Errorf("result for dropped errored call survived: %+v", p)
			}
		}
	}
}

func TestNormalizeIsIdempotent(t *testing.T) {
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "q"),
		assistantTool("anthropic", "x"), // orphan -> gets synthetic result on first pass
		{Role: llm.RoleAssistant, Provider: "anthropic", StopReason: "aborted", Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "partial"},
		}},
	}
	once := llm.NormalizeForProvider(msgs, "anthropic")
	twice := llm.NormalizeForProvider(once, "anthropic")
	if !reflect.DeepEqual(once, twice) {
		t.Errorf("normalize not idempotent:\nonce:  %+v\ntwice: %+v", once, twice)
	}
}

func TestNormalizeDoesNotMutateInput(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "keep me", ThinkingSignature: "sig"},
		}},
	}
	before, _ := json.Marshal(msgs)
	_ = llm.NormalizeForProvider(msgs, "openai") // would degrade thinking -> text in the COPY
	after, _ := json.Marshal(msgs)
	if string(before) != string(after) {
		t.Errorf("input mutated by NormalizeForProvider:\nbefore: %s\nafter:  %s", before, after)
	}
}

func TestNormalizeUnsignedThinkingDegradesEvenSameProvider(t *testing.T) {
	// Same provider but NO signature -> cannot be replayed validly -> degrade to text.
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "unsigned"},
		}},
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")
	for _, p := range out[0].Content {
		if p.Type == llm.ContentThinking {
			t.Errorf("unsigned thinking kept as thinking block: %+v", out[0].Content)
		}
	}
}

func TestNormalizePreservesToolCallThoughtSignature(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "gemini", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "c", ToolName: "f", ThoughtSignature: "gemini-ts"},
		}},
		llm.NewToolResultMessage("c", "ok", false),
	}
	out := llm.NormalizeForProvider(msgs, "gemini")
	var found bool
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolUse && p.ThoughtSignature == "gemini-ts" {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("tool-call ThoughtSignature lost through normalize: %+v", out)
	}
}

func TestNormalizeLargeTranscriptNoPanic(t *testing.T) {
	var msgs []llm.Message
	for i := 0; i < 5000; i++ {
		msgs = append(msgs, llm.NewTextMessage(llm.RoleUser, "x"))
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "y"},
		}})
	}
	if got := llm.NormalizeForProvider(msgs, "anthropic"); len(got) != len(msgs) {
		t.Errorf("large transcript length changed unexpectedly: %d -> %d", len(msgs), len(got))
	}
}

func TestNormalizeDropsToolResultBeforeItsCall(t *testing.T) {
	// Out-of-order transcript: a tool_result appears BEFORE the tool_use it
	// answers. Every provider rejects a result with no preceding call.
	msgs := []llm.Message{
		llm.NewToolResultMessage("c1", "premature", false), // result first
		assistantTool("anthropic", "c1"),                   // its call comes later
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")

	// Invariant: every tool_result must appear AFTER a tool_use with the same id.
	seen := map[string]bool{}
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolUse {
				seen[p.ToolUseID] = true
			}
			if p.Type == llm.ContentToolResult && !seen[p.ToolResultID] {
				t.Errorf("tool_result %q appears before its call: %+v", p.ToolResultID, out)
			}
		}
	}
}

func TestNormalizeDeduplicatesToolUseIDs(t *testing.T) {
	// Two turns each emit a tool call with the SAME id (e.g. Gemini's per-turn
	// synthetic ids colliding across turns), each answered. A handoff to a
	// provider that requires unique ids must not see duplicates.
	msgs := []llm.Message{
		assistantTool("gemini", "call_0_f"),
		llm.NewToolResultMessage("call_0_f", "r1", false),
		assistantTool("gemini", "call_0_f"),
		llm.NewToolResultMessage("call_0_f", "r2", false),
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")

	var callIDs []string
	resultCounts := map[string]int{}
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolUse {
				callIDs = append(callIDs, p.ToolUseID)
			}
			if p.Type == llm.ContentToolResult {
				resultCounts[p.ToolResultID]++
			}
		}
	}
	if len(callIDs) != 2 {
		t.Fatalf("want 2 tool calls, got %v", callIDs)
	}
	if callIDs[0] == callIDs[1] {
		t.Errorf("duplicate tool_use ids not deduplicated: %v", callIDs)
	}
	for _, id := range callIDs {
		if resultCounts[id] != 1 {
			t.Errorf("call %q not paired with exactly one result (got %d): %+v", id, resultCounts[id], out)
		}
	}
}

// BUG 10: a Gemini ThoughtSignature riding on a tool_use must NOT survive a
// handoff to a foreign provider — same invariant the thinking handler enforces
// for ThinkingSignature ("signatures don't transfer"). Kept same-provider.
func TestNormalizeStripsThoughtSignatureCrossProvider(t *testing.T) {
	mk := func() []llm.Message {
		return []llm.Message{
			{Role: llm.RoleAssistant, Provider: "gemini", Content: []llm.ContentPart{
				{Type: llm.ContentToolUse, ToolUseID: "c1", ToolName: "f", ToolInput: map[string]any{}, ThoughtSignature: "gemini-sig-xyz"},
			}},
			llm.NewToolResultMessage("c1", "r", false),
		}
	}
	// Cross-provider: signature must be gone.
	for _, m := range llm.NormalizeForProvider(mk(), "openai") {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolUse && p.ThoughtSignature != "" {
				t.Fatalf("ThoughtSignature %q crossed to a foreign provider", p.ThoughtSignature)
			}
		}
	}
	// Same provider: signature preserved for the round-trip it exists for.
	kept := false
	for _, m := range llm.NormalizeForProvider(mk(), "gemini") {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolUse && p.ThoughtSignature == "gemini-sig-xyz" {
				kept = true
			}
		}
	}
	if !kept {
		t.Fatal("ThoughtSignature was dropped on a same-provider replay")
	}
}

// A thinking block that carries a signature but EMPTY text is what Claude
// 4.7+ return by default (display "omitted"). The API docs say to keep such a
// block when the turn goes back to the same model ("a thinking block with
// empty text is still a block"; dropping it fails the preserved-thinking
// chain check), so it is kept for the same provider. For another provider it
// has nothing to preserve and is dropped. The answer survives either way.
// (Before v0.12.0 it was dropped for the same provider too.)
func TestNormalizeEmptySignedThinking(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "", ThinkingSignature: "sig"},
			{Type: llm.ContentText, Text: "answer"},
		}},
	}
	same := llm.NormalizeForProvider(msgs, "anthropic")
	if len(same[0].Content) != 2 || same[0].Content[0].Type != llm.ContentThinking || same[0].Content[0].ThinkingSignature != "sig" {
		t.Fatalf("same provider: signed empty thinking not kept: %+v", same[0].Content)
	}
	other := llm.NormalizeForProvider(msgs, "gemini")
	for _, p := range other[0].Content {
		if p.Type == llm.ContentThinking || (p.Type == llm.ContentText && p.Text == "") {
			t.Fatalf("other provider: empty thinking survived: %+v", p)
		}
	}
	if len(other[0].Content) != 1 || other[0].Content[0].Text != "answer" {
		t.Fatalf("answer text lost during normalization: %+v", other[0].Content)
	}
}

// NormalizeForProvider must produce unique tool_use IDs and correctly paired
// results even when a literal "dup#1" coexists with a duplicated "dup" (whose
// dedup target is also "dup#1").
func TestNormalizeIDCollisionWithLiteralSuffix(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "dup", ToolName: "f", ToolInput: map[string]any{}},
		}},
		llm.NewToolResultMessage("dup", "r0", false),
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "dup#1", ToolName: "f", ToolInput: map[string]any{}}, // literal collision target
		}},
		llm.NewToolResultMessage("dup#1", "r1", false),
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "dup", ToolName: "f", ToolInput: map[string]any{}}, // duplicate of first "dup"
		}},
		llm.NewToolResultMessage("dup", "r2", false),
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")

	// Collect tool_use IDs and tool_result IDs.
	seen := map[string]int{}
	var resultIDs []string
	for _, m := range out {
		for _, p := range m.Content {
			switch p.Type {
			case llm.ContentToolUse:
				seen[p.ToolUseID]++
			case llm.ContentToolResult:
				resultIDs = append(resultIDs, p.ToolResultID)
			}
		}
	}
	for id, n := range seen {
		if n != 1 {
			t.Fatalf("tool_use ID %q appears %d times — dedup produced a collision", id, n)
		}
	}
	// Every tool_result must reference an existing tool_use ID (no dangling pairing).
	for _, rid := range resultIDs {
		if _, ok := seen[rid]; !ok {
			t.Fatalf("tool_result references %q which is not an emitted tool_use ID", rid)
		}
	}
	// Normalize must be idempotent — a second pass must not stack suffixes.
	out2 := llm.NormalizeForProvider(out, "anthropic")
	if len(out2) != len(out) {
		t.Fatalf("normalize not idempotent: len %d then %d", len(out), len(out2))
	}
}

func thinkingTurn(provider, model string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Provider: provider, Model: model, Content: []llm.ContentPart{
		{Type: llm.ContentThinking, Thinking: "old reasoning", ThinkingSignature: "sig"},
		{Type: llm.ContentText, Text: "answer"},
	}}
}

func hasText(m llm.Message, s string) bool {
	for _, p := range m.Content {
		if p.Text == s || p.Thinking == s {
			return true
		}
	}
	return false
}

func TestNormalizeForModelDropsOtherModelsThinking(t *testing.T) {
	for _, tc := range []struct {
		name            string
		fromP, fromM    string
		toP, toM        string
		keepThinking    bool // verbatim thinking block
		keepAsText      bool // degraded to text (provider rules)
		droppedEntirely bool
	}{
		{"same model", "anthropic", "claude-haiku-5-5", "anthropic", "claude-haiku-5-5", true, false, false},
		{"served snapshot ID", "anthropic", "claude-haiku-4-5-20251001", "anthropic", "claude-haiku-4-5", true, false, false},
		{"same provider, other model", "anthropic", "claude-opus-5-5", "anthropic", "claude-haiku-5-5", false, false, true},
		{"other provider and model", "gemini", "gemini-3.8-flash", "anthropic", "claude-haiku-5-5", false, false, true},
		{"family member is not a snapshot", "openai", "gpt-5", "openai", "gpt-5-mini", false, false, true},
		{"minor version is not a snapshot", "anthropic", "claude-opus-4", "anthropic", "claude-opus-4-8", false, false, true},
		{"no recorded model keeps provider rules", "gemini", "", "anthropic", "claude-haiku-5-5", false, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := llm.NormalizeForModel([]llm.Message{thinkingTurn(tc.fromP, tc.fromM)}, tc.toP, tc.toM)
			if len(out) != 1 || !hasText(out[0], "answer") {
				t.Fatalf("visible answer lost: %+v", out)
			}
			var block, text bool
			for _, p := range out[0].Content {
				if p.Type == llm.ContentThinking {
					block = true
				}
				if p.Type == llm.ContentText && p.Text == "old reasoning" {
					text = true
				}
			}
			if block != tc.keepThinking || text != tc.keepAsText {
				t.Fatalf("thinking block=%v as-text=%v, want %v/%v: %+v", block, text, tc.keepThinking, tc.keepAsText, out[0].Content)
			}
			if tc.droppedEntirely && (block || text) {
				t.Fatalf("other model's reasoning survived: %+v", out[0].Content)
			}
		})
	}
}

func TestNormalizeForModelEmptyModelMatchesNormalizeForProvider(t *testing.T) {
	msgs := []llm.Message{thinkingTurn("gemini", "gemini-3.8-flash")}
	a := llm.NormalizeForModel(msgs, "anthropic", "")
	b := llm.NormalizeForProvider(msgs, "anthropic")
	if len(a) != len(b) || len(a[0].Content) != len(b[0].Content) {
		t.Fatalf("NormalizeForModel(model=\"\") = %+v, NormalizeForProvider = %+v", a, b)
	}
}

func TestNormalizeForProviderRepairsOrphanToolCall(t *testing.T) {
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "do it"),
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "call_1", ToolName: "do", ToolInput: map[string]any{}},
		}},
		// no tool result for call_1
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")

	var found bool
	for _, m := range out {
		for _, p := range m.Content {
			if p.Type == llm.ContentToolResult && p.ToolResultID == "call_1" && p.IsError {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("orphan tool call not repaired with synthetic result; got %+v", out)
	}
}

func TestNormalizeForProviderDropsErroredTurn(t *testing.T) {
	msgs := []llm.Message{
		llm.NewTextMessage(llm.RoleUser, "q"),
		{Role: llm.RoleAssistant, Provider: "anthropic", StopReason: "error", Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "partial"},
		}},
	}
	out := llm.NormalizeForProvider(msgs, "anthropic")
	for _, m := range out {
		if m.Role == llm.RoleAssistant {
			t.Errorf("errored assistant turn was not dropped: %+v", m)
		}
	}
}

func TestNormalizeForProviderKeepsSameProviderThinking(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Provider: "anthropic", Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "t", ThinkingSignature: "sig"},
			{Type: llm.ContentText, Text: "answer"},
		}},
	}
	// Same provider → thinking block kept.
	same := llm.NormalizeForProvider(msgs, "anthropic")
	if same[0].Content[0].Type != llm.ContentThinking {
		t.Errorf("same-provider thinking should be kept, got %+v", same[0].Content)
	}
	// Different provider → degraded to text.
	diff := llm.NormalizeForProvider(msgs, "openai")
	for _, p := range diff[0].Content {
		if p.Type == llm.ContentThinking {
			t.Errorf("cross-provider thinking should be degraded, got %+v", diff[0].Content)
		}
	}
}
