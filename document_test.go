package llm_test

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

// validPDFData is a tiny, non-empty base64 payload ("%PDF-1.4").
const validPDFData = "JVBERi0xLjQ="

func TestValidateDocumentSource(t *testing.T) {
	tests := []struct {
		name    string
		part    llm.ContentPart
		wantErr bool
	}{
		{
			name: "valid pdf",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: validPDFData},
			},
			wantErr: false,
		},
		{
			name:    "nil document",
			part:    llm.ContentPart{Type: llm.ContentDocument},
			wantErr: true,
		},
		{
			name: "empty data",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: ""},
			},
			wantErr: true,
		},
		{
			name: "unsupported media type",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "base64", MediaType: "image/png", Data: validPDFData},
			},
			wantErr: true,
		},
		{
			name: "unsupported source type",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "url", MediaType: "application/pdf", Data: validPDFData},
			},
			wantErr: true,
		},
		{
			name: "non-canonical base64",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: "JVBERi0"},
			},
			wantErr: true,
		},
		{
			name: "mislabeled bytes",
			part: llm.ContentPart{
				Type:     llm.ContentDocument,
				Document: &llm.DocumentSource{Type: "base64", MediaType: "application/pdf", Data: "iVBORw0KGgo="},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := llm.ValidateDocumentSource(tc.part)
			if tc.wantErr && err == nil {
				t.Errorf("ValidateDocumentSource(%s) = nil, want error", tc.name)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("ValidateDocumentSource(%s) = %v, want nil", tc.name, err)
			}
		})
	}
}

func TestNewDocumentMessage(t *testing.T) {
	msg := llm.NewDocumentMessage(llm.RoleUser, "application/pdf", validPDFData)

	if msg.Role != llm.RoleUser {
		t.Errorf("Role = %q, want %q", msg.Role, llm.RoleUser)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
	}
	part := msg.Content[0]
	if part.Type != llm.ContentDocument {
		t.Errorf("Type = %q, want %q", part.Type, llm.ContentDocument)
	}
	if part.Document == nil {
		t.Fatal("Document is nil")
	}
	if part.Document.Type != "base64" {
		t.Errorf("Document.Type = %q, want base64", part.Document.Type)
	}
	if part.Document.MediaType != "application/pdf" {
		t.Errorf("Document.MediaType = %q, want application/pdf", part.Document.MediaType)
	}
	if part.Document.Data != validPDFData {
		t.Errorf("Document.Data = %q, want %q", part.Document.Data, validPDFData)
	}

	// The constructed message must pass validation.
	if err := llm.ValidateDocumentSource(part); err != nil {
		t.Errorf("ValidateDocumentSource on constructed message: %v", err)
	}
}
