package gemini

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestBuildRequestResponseSchema(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gemini-2.5-flash"}, baseURL: defaultGeminiBase, providerName: "gemini"}
	msgs := []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}

	// json_schema → responseMimeType + responseSchema
	apiReq, err := c.buildRequest(llm.Request{
		Messages:       msgs,
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONSchema, Schema: map[string]any{"type": "object"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if apiReq.GenerationConfig.ResponseMimeType != "application/json" {
		t.Errorf("responseMimeType = %q", apiReq.GenerationConfig.ResponseMimeType)
	}
	if apiReq.GenerationConfig.ResponseSchema == nil {
		t.Error("responseSchema not set for json_schema")
	}

	// json_object → mime type only, no schema
	apiReq2, _ := c.buildRequest(llm.Request{
		Messages:       msgs,
		ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject},
	})
	if apiReq2.GenerationConfig.ResponseMimeType != "application/json" {
		t.Errorf("json_object responseMimeType = %q", apiReq2.GenerationConfig.ResponseMimeType)
	}
	if apiReq2.GenerationConfig.ResponseSchema != nil {
		t.Errorf("json_object should not set responseSchema, got %+v", apiReq2.GenerationConfig.ResponseSchema)
	}

	// nil → no JSON config
	apiReq3, _ := c.buildRequest(llm.Request{Messages: msgs})
	if apiReq3.GenerationConfig.ResponseMimeType != "" {
		t.Errorf("expected no responseMimeType, got %q", apiReq3.GenerationConfig.ResponseMimeType)
	}
}
