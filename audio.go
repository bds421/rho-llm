package llm

import (
	"bytes"
	"strings"
)

// MaxTranscriptionPromptRunes bounds TranscriptionRequest.Prompt. A prompt is a
// short vocabulary/context hint, not a document; providers cap it (Whisper
// keeps only the final 224 tokens) and an unbounded field invites abuse.
const MaxTranscriptionPromptRunes = 1000

// MaxTranscriptionVocabularyTerms and MaxTranscriptionVocabularyTermRunes bound
// TranscriptionRequest.Vocabulary. Gemini Transcribe accepts up to 1,000 terms
// (best results at 100 or fewer); a term is a word or short phrase.
const (
	MaxTranscriptionVocabularyTerms     = 1000
	MaxTranscriptionVocabularyTermRunes = 100
)

// TranscriptionVocabularyHint renders vocabulary as one line for adapters that
// take free text (Whisper prompt, chat-model instruction).
func TranscriptionVocabularyHint(vocabulary []string) string {
	terms := make([]string, 0, len(vocabulary))
	for _, term := range vocabulary {
		if term = strings.TrimSpace(term); term != "" {
			terms = append(terms, term)
		}
	}
	return strings.Join(terms, ", ")
}

// AudioMediaTypeFromSignature sniffs the container of an audio payload from
// its leading bytes and returns the canonical media type, or "" when the bytes
// match no supported audio format. Adapters compare it with the declared
// media type so a mislabelled or non-audio upload fails before dispatch.
func AudioMediaTypeFromSignature(data []byte) string {
	switch {
	case len(data) >= 4 && bytes.Equal(data[:4], []byte("fLaC")):
		return "audio/flac"
	case len(data) >= 12 && bytes.Equal(data[4:8], []byte("ftyp")):
		return "audio/mp4"
	case len(data) >= 3 && bytes.Equal(data[:3], []byte("ID3")):
		return "audio/mpeg"
	case len(data) >= 2 && data[0] == 0xff && data[1]&0xf6 == 0xf0:
		// ADTS AAC: 12-bit sync with layer bits 00, which MPEG audio reserves.
		return "audio/aac"
	case len(data) >= 2 && data[0] == 0xff && data[1]&0xe0 == 0xe0:
		return "audio/mpeg"
	case len(data) >= 4 && bytes.Equal(data[:4], []byte("OggS")):
		return "audio/ogg"
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WAVE")):
		return "audio/wav"
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("FORM")) &&
		(bytes.Equal(data[8:12], []byte("AIFF")) || bytes.Equal(data[8:12], []byte("AIFC"))):
		return "audio/aiff"
	case len(data) >= 4 && bytes.Equal(data[:4], []byte{0x1a, 0x45, 0xdf, 0xa3}):
		return "audio/webm"
	default:
		return ""
	}
}
