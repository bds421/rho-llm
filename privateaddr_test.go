package llm_test

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/bds421/rho-llm"
)

func TestCheckBaseURLRefusesIPv6FormsOfPrivateAddresses(t *testing.T) {
	refused := []string{
		"http://[64:ff9b::a9fe:a9fe]/v1",
		"http://[64:ff9b::169.254.169.254]/v1",
		"http://[64:ff9b::7f00:1]/v1",
		"http://[64:ff9b::a00:5]/v1",
		"http://[64:ff9b:1::a9fe:a9fe]/v1",
		"http://[2002:a9fe:a9fe::1]/v1",
		"http://[2002:7f00:1::]/v1",
		"http://[::a9fe:a9fe]/v1",
		"http://[::127.0.0.1]/v1",
		"http://[::ffff:169.254.169.254]/v1",
		"http://[::ffff:7f00:1]/v1",
		"http://[2001:0:4136:e378:8000:63bf:3fff:fdd2]/v1",
		"http://[fd00:ec2::254]/v1",
		"http://[fc00::1]/v1",
		"http://[fe80::1]/v1",
		"http://[fe80::1%25en0]/v1",
		"http://[fec0::1]/v1",
		"http://[ff02::1]/v1",
		"http://[::]/v1",
		"http://[::1]:8080/v1",
		"http://[2001:db8::1]/v1",
		"http://0.0.0.0/v1",
		"http://100.64.0.1/v1",
		"http://169.254.169.254./latest",
	}
	for _, baseURL := range refused {
		err := llm.CheckBaseURL(llm.Config{BaseURL: baseURL, BlockPrivateBaseURL: true})
		if !errors.Is(err, llm.ErrPrivateAddress) {
			t.Errorf("CheckBaseURL(%q) = %v, want ErrPrivateAddress", baseURL, err)
		}
	}
}

func TestCheckBaseURLAllowsPublicAddressesAndNames(t *testing.T) {
	allowed := []string{
		"https://[64:ff9b::808:808]/v1",
		"https://[64:ff9b::8.8.8.8]/v1",
		"https://[2606:4700:4700::1111]/v1",
		"https://8.8.8.8/v1",
		"https://api.anthropic.com",
	}
	for _, baseURL := range allowed {
		if err := llm.CheckBaseURL(llm.Config{BaseURL: baseURL, BlockPrivateBaseURL: true}); err != nil {
			t.Errorf("CheckBaseURL(%q) = %v, want nil", baseURL, err)
		}
	}
}

func privateAddrServer(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		_, _ = io.WriteString(w, "ok")
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func privateAddrGet(t *testing.T, client *http.Client, target string) error {
	t.Helper()
	resp, err := client.Get(target)
	if err != nil {
		return err
	}
	_ = resp.Body.Close()
	return nil
}

func TestBlockPrivateBaseURLRefusesResolvedPrivateAddressAtDial(t *testing.T) {
	srv, hits := privateAddrServer(t)
	byName := strings.Replace(srv.URL, "127.0.0.1", "localhost", 1)
	for _, target := range []string{srv.URL, byName} {
		client, err := llm.NewSafeHTTPClient(llm.Config{BlockPrivateBaseURL: true, DisableProxy: true})
		if err != nil {
			t.Fatal(err)
		}
		err = privateAddrGet(t, client, target)
		if !errors.Is(err, llm.ErrPrivateAddress) {
			t.Errorf("GET %s = %v, want ErrPrivateAddress", target, err)
		}
		if llm.IsRetryable(err) {
			t.Errorf("GET %s: refused dial is retryable", target)
		}
	}
	if n := hits.Load(); n != 0 {
		t.Fatalf("server reached %d times despite BlockPrivateBaseURL", n)
	}
}

func TestBlockPrivateBaseURLOffDialsPrivateAddress(t *testing.T) {
	srv, hits := privateAddrServer(t)
	client, err := llm.NewSafeHTTPClient(llm.Config{DisableProxy: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := privateAddrGet(t, client, srv.URL); err != nil {
		t.Fatalf("GET without BlockPrivateBaseURL: %v", err)
	}
	if hits.Load() != 1 {
		t.Fatal("server not reached without BlockPrivateBaseURL")
	}
}

func TestBlockPrivateBaseURLAllowsPrivateProxy(t *testing.T) {
	proxy, hits := privateAddrServer(t)
	client, err := llm.NewSafeHTTPClient(llm.Config{BlockPrivateBaseURL: true, ProxyURL: proxy.URL})
	if err != nil {
		t.Fatal(err)
	}
	if err := privateAddrGet(t, client, "http://8.8.8.8/v1/models"); err != nil {
		t.Fatalf("GET through a private proxy: %v", err)
	}
	if hits.Load() != 1 {
		t.Fatal("request did not go through the proxy")
	}
}

func TestBlockPrivateBaseURLIgnoresAmbientProxy(t *testing.T) {
	proxy, proxyHits := privateAddrServer(t)
	target, targetHits := privateAddrServer(t)
	t.Setenv("HTTP_PROXY", proxy.URL)
	t.Setenv("HTTPS_PROXY", proxy.URL)
	t.Setenv("NO_PROXY", "")
	client, err := llm.NewSafeHTTPClient(llm.Config{BlockPrivateBaseURL: true})
	if err != nil {
		t.Fatal(err)
	}
	if client.Transport.(*http.Transport).Proxy != nil {
		t.Fatal("BlockPrivateBaseURL client uses the ambient proxy")
	}
	byName := strings.Replace(target.URL, "127.0.0.1", "localhost", 1)
	for _, u := range []string{target.URL, byName} {
		if err := privateAddrGet(t, client, u); !errors.Is(err, llm.ErrPrivateAddress) {
			t.Errorf("GET %s = %v, want ErrPrivateAddress", u, err)
		}
	}
	if proxyHits.Load() != 0 || targetHits.Load() != 0 {
		t.Fatalf("proxy hits %d, target hits %d, want 0 and 0", proxyHits.Load(), targetHits.Load())
	}
}

func TestBlockPrivateBaseURLChecksTargetBeforeExplicitProxy(t *testing.T) {
	proxy, proxyHits := privateAddrServer(t)
	target, _ := privateAddrServer(t)
	client, err := llm.NewSafeHTTPClient(llm.Config{BlockPrivateBaseURL: true, ProxyURL: proxy.URL})
	if err != nil {
		t.Fatal(err)
	}
	refused := []string{
		target.URL,
		strings.Replace(target.URL, "127.0.0.1", "localhost", 1),
		"http://app.localhost/v1",
		"http://169.254.169.254/latest",
		"http://[64:ff9b::a9fe:a9fe]/latest",
		"http://[fd00:ec2::254]/latest",
		proxy.URL,
	}
	for _, u := range refused {
		err := privateAddrGet(t, client, u)
		if !errors.Is(err, llm.ErrPrivateAddress) {
			t.Errorf("GET %s via proxy = %v, want ErrPrivateAddress", u, err)
		}
		if llm.IsRetryable(err) {
			t.Errorf("GET %s via proxy: refusal is retryable", u)
		}
	}
	if n := proxyHits.Load(); n != 0 {
		t.Fatalf("proxy received %d refused requests", n)
	}
}

func TestBlockPrivateBaseURLChecksRedirectHopBeforeProxy(t *testing.T) {
	target, targetHits := privateAddrServer(t)
	var proxyHits atomic.Int32
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxyHits.Add(1)
		http.Redirect(w, r, target.URL+"/internal", http.StatusFound)
	}))
	t.Cleanup(proxy.Close)
	client, err := llm.NewSafeHTTPClient(llm.Config{BlockPrivateBaseURL: true, ProxyURL: proxy.URL})
	if err != nil {
		t.Fatal(err)
	}
	if err := privateAddrGet(t, client, "http://8.8.8.8/start"); !errors.Is(err, llm.ErrPrivateAddress) {
		t.Errorf("redirect to %s = %v, want ErrPrivateAddress", target.URL, err)
	}
	if proxyHits.Load() != 1 || targetHits.Load() != 0 {
		t.Fatalf("proxy hits %d, target hits %d, want 1 and 0", proxyHits.Load(), targetHits.Load())
	}
}
