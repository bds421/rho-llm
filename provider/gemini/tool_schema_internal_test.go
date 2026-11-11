package gemini

import (
	"encoding/json"
	"testing"

	llm "github.com/bds421/rho-llm"
)

// Gemini's "parameters" field takes only its OpenAPI subset: a tool schema with
// additionalProperties answered 400 `Unknown name "additionalProperties" at
// 'tools[0].function_declarations[0].parameters'` (live, tutorial 23 on
// v0.15.0). parametersJsonSchema takes full JSON Schema (ai.google.dev/api,
// FunctionDeclaration), so the schema must ride there, unchanged.
func TestToolSchemaSentAsParametersJSONSchema(t *testing.T) {
	schema := map[string]any{
		"$schema":              "https://json-schema.org/draft/2020-12/schema",
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"item"},
		"properties": map[string]any{
			"item": map[string]any{"$ref": "#/$defs/item"},
		},
		"$defs": map[string]any{
			"item": map[string]any{"type": "string", "enum": []any{"a", "b"}},
		},
	}
	c := &Client{config: llm.Config{Model: "gemini-3.8-flash"}, baseURL: defaultGeminiBase, providerName: "gemini"}
	req := llm.Request{
		Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "x")},
		Tools: []llm.Tool{
			{Name: "pick", Description: "pick one", InputSchema: schema},
			{Name: "noargs", Description: "no schema"},
		},
	}
	apiReq, err := c.buildRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(apiReq)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Tools []struct {
			FunctionDeclarations []map[string]json.RawMessage `json:"functionDeclarations"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Tools) != 1 || len(wire.Tools[0].FunctionDeclarations) != 2 {
		t.Fatalf("tools on the wire: %s", data)
	}
	decls := wire.Tools[0].FunctionDeclarations
	for _, d := range decls {
		if _, ok := d["parameters"]; ok {
			t.Fatalf("declaration still uses the OpenAPI-subset field \"parameters\": %s", data)
		}
	}
	var got map[string]any
	if err := json.Unmarshal(decls[0]["parametersJsonSchema"], &got); err != nil {
		t.Fatalf("pick: parametersJsonSchema missing or invalid: %s", data)
	}
	want, _ := json.Marshal(schema)
	gotJSON, _ := json.Marshal(got)
	if string(gotJSON) != string(want) {
		t.Fatalf("schema changed on the way out:\n got %s\nwant %s", gotJSON, want)
	}
	var empty map[string]any
	if err := json.Unmarshal(decls[1]["parametersJsonSchema"], &empty); err != nil || empty["type"] != "object" || len(empty) != 1 {
		t.Fatalf("noargs: want {\"type\":\"object\"}, got %s", decls[1]["parametersJsonSchema"])
	}
}
