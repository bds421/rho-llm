package llm_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestConversationToRequestEmpty(t *testing.T) {
	c := llm.NewConversation("")
	req := c.ToRequest(llm.Request{Model: "m"})
	if len(req.Messages) != 0 || req.Model != "m" {
		t.Errorf("unexpected request from empty conversation: %+v", req)
	}
}

func TestLoadConversationMalformed(t *testing.T) {
	cases := []struct {
		name    string
		data    string
		wantErr bool
	}{
		{"empty", ``, true},
		{"garbage", `not json at all`, true},
		{"array not object", `[]`, true},
		{"truncated", `{"messages":[{"role":"user"`, true},
		{"null", `null`, false},                              // -> empty conversation, version normalized
		{"empty object", `{}`, false},                        // -> version normalized to 1
		{"negative version", `{"schema_version":-7}`, false}, // -> normalized to 1
		{"null messages", `{"schema_version":1,"messages":null}`, false},
		{"future version", `{"schema_version":999999}`, true}, // newer than build -> reject
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := llm.LoadConversation([]byte(tc.data))
			if tc.wantErr && err == nil {
				t.Errorf("expected error, got conv %+v", c)
			}
			if !tc.wantErr {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				} else if c.SchemaVersion < 1 {
					t.Errorf("schema_version not normalized: %d", c.SchemaVersion)
				}
			}
		})
	}
}

func TestConversationRoundTripAllContentTypes(t *testing.T) {
	conv := llm.NewConversation("sys",
		llm.Tool{Name: "t", Description: "d", InputSchema: map[string]any{"type": "object"}})
	conv.Append(
		llm.Message{Role: llm.RoleUser, Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "look"},
			{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}},
			{Type: llm.ContentDocument, Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: "JVBE"}},
		}},
		llm.Message{Role: llm.RoleAssistant, Provider: "anthropic", Model: "claude-sonnet-4-6", StopReason: "tool_use", Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "reason", ThinkingSignature: "sig"},
			{Type: llm.ContentThinking, Thinking: "enc", Redacted: true},
			{Type: llm.ContentText, Text: "calling"},
			{Type: llm.ContentToolUse, ToolUseID: "c1", ToolName: "t", ToolInput: map[string]any{"x": 1.0}, ThoughtSignature: "ts"},
		}},
		llm.NewToolResultMessage("c1", "42", false),
	)

	blob, err := json.Marshal(conv)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := llm.LoadConversation(blob)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !reflect.DeepEqual(conv.Messages, got.Messages) {
		t.Errorf("messages changed across round-trip:\n in: %+v\nout: %+v", conv.Messages, got.Messages)
	}
}

// Clone()'s primary path is a JSON round-trip. A non-JSON-serializable ToolInput
// (a channel) forces the fallback path. The fallback must STILL produce an
// isolated copy — mutating the clone must not corrupt the original.
func TestCloneFallbackIsolatesNestedStateOnUnserializableInput(t *testing.T) {
	conv := llm.NewConversation("sys", llm.Tool{
		Name:        "t",
		InputSchema: map[string]any{"type": "object"},
	})
	conv.Append(llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentPart{
			{Type: llm.ContentText, Text: "original"},
			// chan is not JSON-serializable → forces Clone's fallback path
			{Type: llm.ContentToolUse, ToolUseID: "c1", ToolName: "t", ToolInput: make(chan int)},
		},
	})
	conv.Append(llm.NewToolResultParts("c1", false,
		llm.ContentPart{Type: llm.ContentText, Text: "nested-original"},
	))

	cp := conv.Clone()

	// Mutate every nesting level of the clone.
	cp.Messages[0].Content[0].Text = "MUTATED"
	cp.Messages[1].Content[0].ToolResultParts[0].Text = "MUTATED"
	cp.Tools[0].InputSchema["type"] = "MUTATED"
	cp.System = "MUTATED"

	if got := conv.Messages[0].Content[0].Text; got != "original" {
		t.Errorf("clone shares Content backing array: original text corrupted to %q", got)
	}
	if got := conv.Messages[1].Content[0].ToolResultParts[0].Text; got != "nested-original" {
		t.Errorf("clone shares nested ToolResultParts: corrupted to %q", got)
	}
	if got := conv.Tools[0].InputSchema["type"]; got != "object" {
		t.Errorf("clone shares Tool.InputSchema map: corrupted to %v", got)
	}
	if conv.System != "sys" {
		t.Errorf("clone shares top-level fields: System corrupted to %q", conv.System)
	}
}

// Clone of an image/document message must not share the *ImageSource /
// *DocumentSource pointers.
func TestCloneFallbackIsolatesPointerSources(t *testing.T) {
	conv := llm.NewConversation("")
	conv.Append(llm.Message{
		Role: llm.RoleUser,
		Content: []llm.ContentPart{
			{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAA"}},
			// chan forces the fallback path
			{Type: llm.ContentToolUse, ToolInput: make(chan int)},
		},
	})
	cp := conv.Clone()
	cp.Messages[0].Content[0].Source.Data = "MUTATED"
	if got := conv.Messages[0].Content[0].Source.Data; got != "AAA" {
		t.Fatalf("clone shares *ImageSource pointer: corrupted to %q", got)
	}
}

func TestLoadConversationHostileBlobs(t *testing.T) {
	cases := map[string]string{
		"schema_version as string": `{"schema_version":"99","messages":[]}`,
		"messages as object":       `{"schema_version":1,"messages":{"not":"an array"}}`,
		"content as string":        `{"schema_version":1,"messages":[{"role":"user","content":"should be array"}]}`,
		"truncated":                `{"schema_version":1,"messages":[{"role":"user",`,
		"trailing garbage":         `{"schema_version":1,"messages":[]} JUNK`,
		"null bytes":               "{\"schema_version\":1,\x00\"messages\":[]}",
		"future schema":            `{"schema_version":999999,"messages":[]}`,
		"deeply nested tool input": `{"schema_version":1,"messages":[{"role":"assistant","content":[{"type":"tool_use","input":` + strings.Repeat(`[`, 2000) + strings.Repeat(`]`, 2000) + `}]}]}`,
	}
	for name, blob := range cases {
		t.Run(name, func(t *testing.T) {
			// Must not panic; valid blobs return a conv, invalid ones an error.
			conv, err := llm.LoadConversation([]byte(blob))
			if err == nil && conv == nil {
				t.Fatal("both conv and err are nil")
			}
		})
	}
}

// BUG 3: a circular ToolInput makes json.Marshal fail (cycle detected), forcing
// Clone's structural fallback, whose deepCopyJSONValue recursed with no cycle/
// depth guard — a stack-overflow crash (not even recoverable). Clone must return
// instead. (A real JSON-sourced ToolInput can't be cyclic; only hand-built Go
// maps reach here — but a library deep-copy must never crash on its input.)
func TestConversationCloneCircularToolInputDoesNotCrash(t *testing.T) {
	conv := llm.NewConversation("sys")
	circular := map[string]any{"k": "v"}
	circular["self"] = circular // cycle
	conv.Append(llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentPart{
		{Type: llm.ContentToolUse, ToolUseID: "c1", ToolName: "t", ToolInput: circular},
	}})

	cp := conv.Clone() // must not stack-overflow
	if cp == nil {
		t.Fatal("Clone returned nil")
	}
	if len(cp.Messages) != 1 {
		t.Fatalf("got %d messages, want 1", len(cp.Messages))
	}
}

func TestConversationSerializationRoundTrip(t *testing.T) {
	conv := llm.NewConversation("be terse",
		llm.Tool{Name: "get_weather", Description: "weather", InputSchema: map[string]any{"type": "object"}},
	)
	conv.Append(llm.NewTextMessage(llm.RoleUser, "hi"))
	conv.Append(llm.Message{
		Role:     llm.RoleAssistant,
		Provider: "anthropic",
		Model:    "claude-sonnet-4-6",
		Content: []llm.ContentPart{
			{Type: llm.ContentThinking, Thinking: "hmm", ThinkingSignature: "sig123"},
			{Type: llm.ContentText, Text: "hello"},
		},
	})

	data, err := json.Marshal(conv)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	got, err := llm.LoadConversation(data)
	if err != nil {
		t.Fatalf("LoadConversation: %v", err)
	}
	if got.SchemaVersion != llm.ConversationSchemaVersion {
		t.Errorf("SchemaVersion = %d, want %d", got.SchemaVersion, llm.ConversationSchemaVersion)
	}
	if got.System != "be terse" {
		t.Errorf("System = %q", got.System)
	}
	if len(got.Tools) != 1 || got.Tools[0].Name != "get_weather" {
		t.Errorf("Tools not round-tripped: %+v", got.Tools)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("len(Messages) = %d, want 2", len(got.Messages))
	}
	am := got.Messages[1]
	if am.Provider != "anthropic" || am.Model != "claude-sonnet-4-6" {
		t.Errorf("provenance lost: provider=%q model=%q", am.Provider, am.Model)
	}
	if len(am.Content) != 2 || am.Content[0].Type != llm.ContentThinking {
		t.Fatalf("content blocks not round-tripped: %+v", am.Content)
	}
	if am.Content[0].ThinkingSignature != "sig123" {
		t.Errorf("thinking signature lost: %q", am.Content[0].ThinkingSignature)
	}
}

func TestConversationToRequestPreservesBase(t *testing.T) {
	conv := llm.NewConversation("sys")
	conv.Append(llm.NewTextMessage(llm.RoleUser, "q"))

	base := llm.Request{Model: "claude-sonnet-4-6", MaxTokens: 1234, ThinkingLevel: llm.ThinkingLow}
	req := conv.ToRequest(base)

	if req.Model != "claude-sonnet-4-6" || req.MaxTokens != 1234 || req.ThinkingLevel != llm.ThinkingLow {
		t.Errorf("base fields not preserved: %+v", req)
	}
	if req.System != "sys" {
		t.Errorf("System not set from conversation: %q", req.System)
	}
	if len(req.Messages) != 1 {
		t.Errorf("Messages not set from conversation: %d", len(req.Messages))
	}
}

func TestConversationAddResponseAccumulatesUsage(t *testing.T) {
	conv := llm.NewConversation("")
	r1 := &llm.Response{Model: "claude-sonnet-4-6", Content: "a", InputTokens: 10, OutputTokens: 5}
	r2 := &llm.Response{Model: "claude-sonnet-4-6", Content: "b", InputTokens: 20, OutputTokens: 7}

	conv.AddResponse("anthropic", r1)
	conv.AddResponse("anthropic", r2)

	if len(conv.Messages) != 2 {
		t.Fatalf("len(Messages) = %d, want 2", len(conv.Messages))
	}
	if conv.Messages[0].Provider != "anthropic" {
		t.Errorf("provenance not set: %q", conv.Messages[0].Provider)
	}
	if conv.Usage.InputTokens != 30 || conv.Usage.OutputTokens != 12 {
		t.Errorf("usage not accumulated: in=%d out=%d", conv.Usage.InputTokens, conv.Usage.OutputTokens)
	}
	if conv.Usage.Cost < 0 {
		t.Errorf("cost should be non-negative, got %v", conv.Usage.Cost)
	}
}

func TestLoadConversationRejectsNewerSchema(t *testing.T) {
	data := []byte(`{"schema_version": 9999, "messages": []}`)
	if _, err := llm.LoadConversation(data); err == nil {
		t.Error("expected error for newer schema_version, got nil")
	}
}

func TestMarshalStampsSchemaVersionOnZeroValue(t *testing.T) {
	var conv llm.Conversation // zero value, SchemaVersion == 0
	data, err := json.Marshal(conv)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw struct {
		SchemaVersion int `json:"schema_version"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw.SchemaVersion != llm.ConversationSchemaVersion {
		t.Errorf("schema_version = %d, want %d", raw.SchemaVersion, llm.ConversationSchemaVersion)
	}
}

// The serialized form must round-trip losslessly even for hostile content:
// unicode, emoji, embedded control chars (incl. NUL), and every content kind.
// Invariant: marshal -> load -> marshal is idempotent (no field dropped/mangled).
func TestConversationRoundTripHostileContent(t *testing.T) {
	conv := llm.NewConversation("sys \x00\n\t 你好 🤖 \"quoted\" </script>")
	conv.Append(
		llm.NewTextMessage(llm.RoleUser, "emoji 🤖🔥 ctl \x01\x02 nl\nrtl ‮ wj‍"),
		llm.Message{Role: llm.RoleAssistant, Provider: "anthropic", Model: "claude-sonnet-4-6", StopReason: llm.StopToolUse,
			Content: []llm.ContentPart{
				{Type: llm.ContentThinking, Thinking: "reason 你好", ThinkingSignature: "sig=="},
				{Type: llm.ContentToolUse, ToolUseID: "c1\x00x", ToolName: "f", ToolInput: map[string]any{
					"k": "v\x00", "nested": map[string]any{"a": []any{"x", 1.5, true, nil}}, "uni": "🤖",
				}},
				{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA=="}},
			}},
		llm.NewToolResultMessage("c1\x00x", "result 🤖", true),
	)

	b1, err := json.Marshal(conv)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	loaded, err := llm.LoadConversation(b1)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	b2, err := json.Marshal(loaded)
	if err != nil {
		t.Fatalf("re-marshal: %v", err)
	}
	if string(b1) != string(b2) {
		t.Fatalf("round-trip not idempotent — a field was dropped/mangled:\n  first:  %s\n  second: %s", b1, b2)
	}
}

// R-M1: Conversation.Clone must deep-copy every nesting level the serialized
// form can carry — content parts, tool-result sub-parts, and tool inputs.
func TestConversationCloneIsDeep(t *testing.T) {
	conv := llm.NewConversation("sys")
	conv.Append(llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentPart{
			{Type: llm.ContentToolUse, ToolUseID: "c1", ToolName: "t", ToolInput: map[string]any{"k": "v"}},
		},
	})
	conv.Append(llm.NewToolResultParts("c1", false,
		llm.ContentPart{Type: llm.ContentText, Text: "nested"},
	))

	cp := conv.Clone()
	cp.Messages[0].Content[0].ToolInput.(map[string]any)["k"] = "TAMPERED"
	cp.Messages[1].Content[0].ToolResultParts[0].Text = "TAMPERED"
	cp.System = "TAMPERED"

	if got := conv.Messages[0].Content[0].ToolInput.(map[string]any)["k"]; got != "v" {
		t.Fatalf("Clone shares tool input map: %v", got)
	}
	if got := conv.Messages[1].Content[0].ToolResultParts[0].Text; got != "nested" {
		t.Fatalf("Clone shares nested tool-result parts: %q", got)
	}
	if conv.System != "sys" {
		t.Fatal("Clone shares top-level fields")
	}
}
