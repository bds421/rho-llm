package llm

import (
	"context"
	"errors"
	"net/http"
	"net/netip"
	"testing"
)

func TestCheckProxiedTargetRefusesTheProxyItself(t *testing.T) {
	for _, target := range []string{"http://8.8.8.8:3128/", "http://8.8.8.8:3128/v1/models"} {
		req, err := http.NewRequest(http.MethodGet, target, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := checkProxiedTarget(req, "8.8.8.8:3128"); !errors.Is(err, ErrPrivateAddress) {
			t.Errorf("checkProxiedTarget(%s) = %v, want ErrPrivateAddress", target, err)
		}
	}
	req, err := http.NewRequest(http.MethodGet, "http://8.8.8.8/", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkProxiedTarget(req, "8.8.8.8:3128"); err != nil {
		t.Errorf("checkProxiedTarget(other port) = %v, want nil", err)
	}
}

func TestCheckProxiedTargetResolvesNames(t *testing.T) {
	answers := map[string][]netip.Addr{
		"metadata.example": {netip.MustParseAddr("169.254.169.254")},
		"mixed.example":    {netip.MustParseAddr("8.8.8.8"), netip.MustParseAddr("fd00::1")},
		"nat64.example":    {netip.MustParseAddr("64:ff9b::a00:1")},
		"public.example":   {netip.MustParseAddr("8.8.8.8"), netip.MustParseAddr("64:ff9b::808:808")},
	}
	original := lookupNetIP
	lookupNetIP = func(_ context.Context, _, host string) ([]netip.Addr, error) { return answers[host], nil }
	t.Cleanup(func() { lookupNetIP = original })
	for host, wantRefused := range map[string]bool{
		"metadata.example": true, "mixed.example": true, "nat64.example": true, "public.example": false,
	} {
		req, err := http.NewRequest(http.MethodGet, "https://"+host+"/v1", nil)
		if err != nil {
			t.Fatal(err)
		}
		err = checkProxiedTarget(req, "10.0.0.1:3128")
		if refused := errors.Is(err, ErrPrivateAddress); refused != wantRefused {
			t.Errorf("checkProxiedTarget(%s) = %v, want refused=%v", host, err, wantRefused)
		}
	}
}
