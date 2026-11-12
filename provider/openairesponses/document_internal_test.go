package openairesponses

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/bds421/rho-llm"
)

const testPDFData = "JVBERi0xLjQ=" // base64("%PDF-1.4")

// A PDF is sent as an input_file item with the bytes as a data URL. The
// adapter refused documents, so a session that had read a PDF on another
// provider failed every turn after a handoff to the Responses API (B20).
func TestBuildRequestSendsPDFAsInputFile(t *testing.T) {
	c := &Client{config: llm.Config{Model: "gpt-5.1"}, providerName: "openai_responses"}
	pdf := llm.ContentPart{Type: llm.ContentDocument, Title: "report.pdf",
		Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: testPDFData}}
	untitled := llm.ContentPart{Type: llm.ContentDocument,
		Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: testPDFData}}
	req := llm.Request{MaxTokens: 1024, Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentPart{
		{Type: llm.ContentText, Text: "summarize"}, pdf, untitled,
	}}}}
	apiReq, err := c.buildRequest(req, false)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(apiReq.Input)
	for _, want := range []string{
		`{"text":"summarize","type":"input_text"}`,
		`"file_data":"data:application/pdf;base64,` + testPDFData + `","filename":"report.pdf","type":"input_file"`,
		`"filename":"document-2.pdf"`,
	} {
		if !strings.Contains(string(b), want) {
			t.Errorf("input = %s\nmissing %s", b, want)
		}
	}

	for name, bad := range map[string]llm.ContentPart{
		"text document": llm.NewTextDocument("t", "body"),
		"not a pdf":     {Type: llm.ContentDocument, Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: "aGVsbG8="}},
		"nil source":    {Type: llm.ContentDocument},
	} {
		req.Messages[0].Content = []llm.ContentPart{bad}
		if _, err := c.buildRequest(req, false); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}
