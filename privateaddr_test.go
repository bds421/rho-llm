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
		"http://0x7f.1/v1",
		"http://0X7F000001/v1",
		"http://2130706433/v1",
		"http://0177.0.0.1/v1",
		"http://127.1/v1",
		"http://api.example.0x10/v1",
		"http://app.localhost/v1",
		"http://192.0.2.10/v1",
		"http://198.18.0.1/v1",
		"http://240.0.0.1/v1",
		"http://255.255.255.255/v1",
		"http://224.0.0.1/v1",
		"http://[3fff::1]/v1",
		"http://[::ffff:0:a00:1]/v1",
		"http://[::ffff:100.64.0.1]/v1",
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
	subdomain := strings.Replace(srv.URL, "127.0.0.1", "app.localhost", 1)
	for _, target := range []string{srv.URL, byName, subdomain} {
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

type recordingTransport struct{ hosts []string }

func (r *recordingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r.hosts = append(r.hosts, req.URL.Host)
	return nil, errors.New("recorded")
}

func TestHTTPClientIsUsedThroughACopy(t *testing.T) {
	givenTransport := &recordingTransport{}
	given := &http.Client{Transport: givenTransport}
	got, err := llm.NewSafeHTTPClient(llm.Config{HTTPClient: given, BlockPrivateBaseURL: true})
	if err != nil {
		t.Fatal(err)
	}
	if got == given || got.Transport != given.Transport || given.CheckRedirect != nil {
		t.Fatalf("NewSafeHTTPClient = %p (transport %p), given %p (transport %p, CheckRedirect set: %v); want a copy on the same transport, given unchanged",
			got, got.Transport, given, given.Transport, given.CheckRedirect != nil)
	}
	for _, cfg := range []llm.Config{
		{HTTPClient: given, ProxyURL: "http://proxy.example:3128"},
		{HTTPClient: given, DisableProxy: true},
	} {
		if _, err := llm.NewSafeHTTPClient(cfg); err == nil {
			t.Errorf("NewSafeHTTPClient(%+v) = nil error, want refusal", cfg)
		}
	}

	recorder := &recordingTransport{}
	client, err := llm.NewClient(llm.Config{
		Provider: "openai", Model: "gpt-4.1", APIKey: "k", MaxTokens: 10, DisableRetries: true,
		BaseURL: "https://api.example.com/v1", HTTPClient: &http.Client{Transport: recorder},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = client.Close() }()
	_, _ = client.Complete(t.Context(), llm.Request{Messages: []llm.Message{llm.NewTextMessage(llm.RoleUser, "hi")}})
	if len(recorder.hosts) == 0 || recorder.hosts[0] != "api.example.com" {
		t.Fatalf("caller's client saw %v, want a request to api.example.com", recorder.hosts)
	}
}

func TestDeviceAuthUsesTheConfiguredClient(t *testing.T) {
	srv, hits := privateAddrServer(t)
	cfg := llm.DeviceAuthConfig{ClientID: "c", DeviceAuthURL: srv.URL, TokenURL: srv.URL,
		HTTP: llm.Config{BlockPrivateBaseURL: true}}
	if _, err := llm.StartDeviceAuth(t.Context(), cfg); !errors.Is(err, llm.ErrPrivateAddress) {
		t.Errorf("StartDeviceAuth to %s = %v, want ErrPrivateAddress", srv.URL, err)
	}
	if _, err := llm.PollDeviceToken(t.Context(), cfg, &llm.DeviceAuth{DeviceCode: "d", Interval: 1}); !errors.Is(err, llm.ErrPrivateAddress) {
		t.Errorf("PollDeviceToken to %s = %v, want ErrPrivateAddress", srv.URL, err)
	}
	if hits.Load() != 0 {
		t.Fatal("OAuth endpoint reached despite BlockPrivateBaseURL")
	}

	recorder := &recordingTransport{}
	cfg = llm.DeviceAuthConfig{ClientID: "c", DeviceAuthURL: "https://auth.example.com/device",
		HTTP: llm.Config{HTTPClient: &http.Client{Transport: recorder}}}
	_, _ = llm.StartDeviceAuth(t.Context(), cfg)
	if len(recorder.hosts) != 1 || recorder.hosts[0] != "auth.example.com" {
		t.Fatalf("caller's client saw %v, want one request to auth.example.com", recorder.hosts)
	}
}

func redirectingServer(t *testing.T, to string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, to, http.StatusFound)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestHTTPClientCopyStripsKeysOnCrossOriginRedirect(t *testing.T) {
	var leaked atomic.Value
	leaked.Store("")
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		leaked.Store(r.Header.Get("x-api-key") + r.Header.Get("x-goog-api-key"))
	}))
	t.Cleanup(target.Close)
	origin := redirectingServer(t, target.URL+"/next")

	var callerCalls atomic.Int32
	given := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error {
		callerCalls.Add(1)
		return nil
	}}
	client, err := llm.NewSafeHTTPClient(llm.Config{HTTPClient: given})
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest(http.MethodGet, origin.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("x-api-key", "secret-a")
	req.Header.Set("x-goog-api-key", "secret-b")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if got := leaked.Load().(string); got != "" {
		t.Errorf("redirect target received API key headers %q", got)
	}
	if callerCalls.Load() != 1 {
		t.Errorf("caller's CheckRedirect ran %d times, want 1", callerCalls.Load())
	}
}

func TestHTTPClientCopyRefusesPrivateRedirectsWithBlockPrivateBaseURL(t *testing.T) {
	target, hits := privateAddrServer(t)
	byName := strings.Replace(target.URL, "127.0.0.1", "localhost", 1)
	for _, to := range []string{target.URL, byName, "http://169.254.169.254/latest", "http://0x7f.1/", "http://[::ffff:10.0.0.1]/"} {
		origin := redirectingServer(t, to)
		client, err := llm.NewSafeHTTPClient(llm.Config{HTTPClient: &http.Client{}, BlockPrivateBaseURL: true})
		if err != nil {
			t.Fatal(err)
		}
		if err := privateAddrGet(t, client, origin.URL); !errors.Is(err, llm.ErrPrivateAddress) {
			t.Errorf("redirect to %s = %v, want ErrPrivateAddress", to, err)
		}
	}
	if hits.Load() != 0 {
		t.Fatal("private redirect target reached")
	}
}

func TestHTTPClientCopyKeepsTheDefaultRedirectLimit(t *testing.T) {
	var loop *httptest.Server
	loop = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, loop.URL+"/again", http.StatusFound)
	}))
	t.Cleanup(loop.Close)
	client, err := llm.NewSafeHTTPClient(llm.Config{HTTPClient: &http.Client{}})
	if err != nil {
		t.Fatal(err)
	}
	err = privateAddrGet(t, client, loop.URL)
	if err == nil || !strings.Contains(err.Error(), "stopped after 10 redirects") {
		t.Fatalf("redirect loop through the caller's client = %v, want net/http's 10-redirect error", err)
	}
}
