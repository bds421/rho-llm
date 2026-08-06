package llm_test

// Hardening pass 9 — ValidateImageSource direct adversarial coverage.

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

// ValidateImageSource must reject every structurally-malformed image part (the
// document validator had a direct test; the image one did not).
func TestValidateImageSourceRejectsMalformed(t *testing.T) {
	good := llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{Type: "base64", MediaType: "image/png", Data: "iVBORw0KGgo="}}
	if err := llm.ValidateImageSource(good); err != nil {
		t.Fatalf("a valid image part was rejected: %v", err)
	}

	bad := map[string]*llm.ImageSource{
		"nil source":             nil,
		"empty data":             {Type: "base64", MediaType: "image/png", Data: ""},
		"unsupported media type": {Type: "base64", MediaType: "image/tiff", Data: "abc"},
		"empty media type":       {Type: "base64", MediaType: "", Data: "abc"},
		"non-base64 source type": {Type: "url", MediaType: "image/png", Data: "abc"},
		"non-canonical base64":   {Type: "base64", MediaType: "image/png", Data: "iVBORw0KGgo"},
		"unknown signature":      {Type: "base64", MediaType: "image/png", Data: "bm90LWltYWdl"},
		"mislabeled jpeg":        {Type: "base64", MediaType: "image/png", Data: "/9j/4A=="},
	}
	for name, src := range bad {
		part := llm.ContentPart{Type: llm.ContentImage, Source: src}
		if err := llm.ValidateImageSource(part); err == nil {
			t.Errorf("%s: ValidateImageSource accepted malformed input", name)
		}
	}
}

func TestValidateImageSourceAcceptsEverySupportedSignature(t *testing.T) {
	for mediaType, data := range map[string]string{
		"image/png":  "iVBORw0KGgo=",
		"image/jpeg": "/9j/4A==",
		"image/gif":  "R0lGODdh",
		"image/webp": "UklGRgQAAABXRUJQ",
	} {
		part := llm.ContentPart{Type: llm.ContentImage, Source: &llm.ImageSource{
			Type: "base64", MediaType: mediaType, Data: data,
		}}
		if err := llm.ValidateImageSource(part); err != nil {
			t.Errorf("%s: %v", mediaType, err)
		}
	}
}
