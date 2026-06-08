package openaicompat

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestBuildRequestResponseFormat(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-4"}, providerName: "openai"}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}

	// json_object
	apiReq, err := c.buildRequest(llm.Request{Messages: msgs, ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}}, false)
	if err != nil {
		t.Fatal(err)
	}
	rf, ok := apiReq.ResponseFormat.(map[string]any)
	if !ok || rf["type"] != "json_object" {
		t.Errorf("json_object: response_format = %+v", apiReq.ResponseFormat)
	}

	// json_schema
	apiReq2, err := c.buildRequest(llm.Request{
		Messages:       msgs,
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Name: "person", Schema: map[string]any{"type": "object"}},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	rf2, _ := apiReq2.ResponseFormat.(map[string]any)
	if rf2["type"] != "json_schema" {
		t.Fatalf("json_schema: response_format = %+v", apiReq2.ResponseFormat)
	}
	js, _ := rf2["json_schema"].(map[string]any)
	if js["name"] != "person" || js["schema"] == nil {
		t.Errorf("json_schema payload wrong: %+v", js)
	}

	// nil → no response_format
	apiReq3, _ := c.buildRequest(llm.Request{Messages: msgs}, false)
	if apiReq3.ResponseFormat != nil {
		t.Errorf("expected no response_format, got %+v", apiReq3.ResponseFormat)
	}
}
