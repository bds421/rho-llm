package llm_test

import (
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestAudioMediaTypeFromSignature(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
		want string
	}{
		{"wav", []byte("RIFF\x00\x00\x00\x00WAVEfmt "), "audio/wav"},
		{"riff but avi", []byte("RIFF\x00\x00\x00\x00AVI LIST"), ""},
		{"flac", []byte("fLaC\x00\x00"), "audio/flac"},
		{"mp3 id3", []byte("ID3\x04\x00"), "audio/mpeg"},
		{"mp3 frame sync", []byte{0xff, 0xfb, 0x90, 0x00}, "audio/mpeg"},
		{"aac adts mpeg-4", []byte{0xff, 0xf1, 0x50, 0x80}, "audio/aac"},
		{"aac adts mpeg-2", []byte{0xff, 0xf9, 0x50, 0x80}, "audio/aac"},
		{"ogg", []byte("OggS\x00\x02"), "audio/ogg"},
		{"m4a", []byte("\x00\x00\x00\x20ftypM4A "), "audio/mp4"},
		{"aiff", []byte("FORM\x00\x00\x00\x00AIFFCOMM"), "audio/aiff"},
		{"aifc", []byte("FORM\x00\x00\x00\x00AIFCFVER"), "audio/aiff"},
		{"form but not audio", []byte("FORM\x00\x00\x00\x00ILBM"), ""},
		{"webm", []byte{0x1a, 0x45, 0xdf, 0xa3, 0x01}, "audio/webm"},
		{"png", []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}, ""},
		{"text", []byte("hello world, not audio"), ""},
		{"nil", nil, ""},
		{"one byte", []byte{0xff}, ""},
		{"truncated riff", []byte("RIFF\x00\x00"), ""},
		{"truncated form", []byte("FORM\x00\x00\x00\x00AI"), ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := llm.AudioMediaTypeFromSignature(tc.data); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}
