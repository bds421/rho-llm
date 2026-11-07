package llm

import (
	"context"
	"crypto/x509"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

// connectProxy is a CONNECT-only forward proxy that tunnels every CONNECT to
// backend, whatever the requested target, and records the targets.
type connectProxy struct {
	*httptest.Server
	mu      sync.Mutex
	targets []string
	refuse  func(target string) bool
}

func newConnectProxy(t *testing.T, backend string, refuse func(target string) bool) *connectProxy {
	t.Helper()
	p := &connectProxy{refuse: refuse}
	p.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p.mu.Lock()
		p.targets = append(p.targets, r.Method+" "+r.Host)
		p.mu.Unlock()
		if r.Method != http.MethodConnect {
			http.Error(w, "CONNECT only", http.StatusMethodNotAllowed)
			return
		}
		if p.refuse != nil && p.refuse(r.Host) {
			http.Error(w, "unreachable", http.StatusBadGateway)
			return
		}
		upstream, err := net.Dial("tcp", backend)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			_ = upstream.Close()
			return
		}
		_, _ = io.WriteString(conn, "HTTP/1.1 200 Connection established\r\n\r\n")
		go func() { _, _ = io.Copy(upstream, conn); _ = upstream.Close() }()
		go func() { _, _ = io.Copy(conn, upstream); _ = conn.Close() }()
	}))
	t.Cleanup(p.Close)
	return p
}

func (p *connectProxy) seen() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.targets...)
}

// hostRecorder is a backend that records the Host header of each request.
type hostRecorder struct {
	mu    sync.Mutex
	hosts []string
}

func (h *hostRecorder) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	h.hosts = append(h.hosts, r.Host)
	h.mu.Unlock()
	_, _ = io.WriteString(w, "ok")
}

func (h *hostRecorder) seen() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.hosts...)
}

func stubLookup(t *testing.T, answers map[string][]netip.Addr, errs map[string]error) *int {
	t.Helper()
	calls := 0
	original := lookupNetIP
	lookupNetIP = func(_ context.Context, _, host string) ([]netip.Addr, error) {
		calls++
		return answers[host], errs[host]
	}
	t.Cleanup(func() { lookupNetIP = original })
	return &calls
}

func blockingClient(t *testing.T, proxyURL string) *http.Client {
	t.Helper()
	client, err := NewSafeHTTPClient(Config{BlockPrivateBaseURL: true, ProxyURL: proxyURL})
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func fetch(client *http.Client, target string) error {
	resp, err := client.Get(target)
	if err != nil {
		return err
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.Body.Close()
}

func TestProxyTunnelPinsTheCheckedAddressForHTTP(t *testing.T) {
	recorder := &hostRecorder{}
	backend := httptest.NewServer(recorder)
	t.Cleanup(backend.Close)
	proxy := newConnectProxy(t, backend.Listener.Addr().String(), nil)
	calls := stubLookup(t, map[string][]netip.Addr{"api.example": {netip.MustParseAddr("8.8.8.8")}}, nil)

	if err := fetch(blockingClient(t, proxy.URL), "http://api.example/v1/models"); err != nil {
		t.Fatal(err)
	}
	if got := proxy.seen(); len(got) != 1 || got[0] != "CONNECT 8.8.8.8:80" {
		t.Errorf("proxy saw %v, want one CONNECT to the checked IP 8.8.8.8:80", got)
	}
	if got := recorder.seen(); len(got) != 1 || got[0] != "api.example" {
		t.Errorf("backend saw Host %v, want api.example", got)
	}
	if *calls != 1 {
		t.Errorf("resolved %d times, want once", *calls)
	}
}

func TestProxyTunnelPinsTheCheckedAddressForHTTPS(t *testing.T) {
	recorder := &hostRecorder{}
	backend := httptest.NewTLSServer(recorder)
	t.Cleanup(backend.Close)
	proxy := newConnectProxy(t, backend.Listener.Addr().String(), nil)
	stubLookup(t, map[string][]netip.Addr{"example.com": {netip.MustParseAddr("8.8.8.8")}}, nil)

	client := blockingClient(t, proxy.URL)
	roots := x509.NewCertPool()
	roots.AddCert(backend.Certificate())
	client.Transport.(*http.Transport).TLSClientConfig.RootCAs = roots
	if err := fetch(client, "https://example.com/v1/models"); err != nil {
		t.Fatal(err)
	}
	if got := proxy.seen(); len(got) != 1 || got[0] != "CONNECT 8.8.8.8:443" {
		t.Errorf("proxy saw %v, want one CONNECT to the checked IP 8.8.8.8:443", got)
	}
	if got := recorder.seen(); len(got) != 1 || got[0] != "example.com" {
		t.Errorf("backend saw Host %v, want example.com", got)
	}
}

func TestProxyTunnelRefusesPrivateAndUnresolvableTargets(t *testing.T) {
	backend := httptest.NewServer(&hostRecorder{})
	t.Cleanup(backend.Close)
	proxy := newConnectProxy(t, backend.Listener.Addr().String(), nil)
	stubLookup(t, map[string][]netip.Addr{
		"metadata.example": {netip.MustParseAddr("169.254.169.254")},
		"mixed.example":    {netip.MustParseAddr("8.8.8.8"), netip.MustParseAddr("fd00::1")},
		"nat64.example":    {netip.MustParseAddr("64:ff9b::a00:1")},
		"cgnat.example":    {netip.MustParseAddr("100.64.1.1")},
		"empty.example":    {},
	}, map[string]error{"broken.example": errors.New("SERVFAIL")})

	client := blockingClient(t, proxy.URL)
	refused := []string{
		backend.URL,
		strings.Replace(backend.URL, "127.0.0.1", "localhost", 1),
		"http://app.localhost/",
		"http://169.254.169.254/latest",
		"http://[64:ff9b::a9fe:a9fe]/latest",
		"http://[fd00:ec2::254]/latest",
		"http://0x7f.1/",
		"http://2130706433/",
		"http://0177.0.0.1/",
		"http://metadata.example/",
		"http://mixed.example/",
		"http://nat64.example/",
		"http://cgnat.example/",
		"http://empty.example/",
		"http://broken.example/",
		proxy.URL,
	}
	for _, target := range refused {
		err := fetch(client, target)
		if !errors.Is(err, ErrPrivateAddress) {
			t.Errorf("GET %s via proxy = %v, want ErrPrivateAddress", target, err)
		}
		if IsRetryable(err) {
			t.Errorf("GET %s via proxy: refusal is retryable", target)
		}
	}
	if got := proxy.seen(); len(got) != 0 {
		t.Fatalf("proxy received refused requests: %v", got)
	}
}

func TestProxyTunnelChecksRedirectHops(t *testing.T) {
	recorder := &hostRecorder{}
	target := httptest.NewServer(recorder)
	t.Cleanup(target.Close)
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+"/internal", http.StatusFound)
	}))
	t.Cleanup(redirector.Close)
	proxy := newConnectProxy(t, redirector.Listener.Addr().String(), nil)

	if err := fetch(blockingClient(t, proxy.URL), "http://8.8.8.8/start"); !errors.Is(err, ErrPrivateAddress) {
		t.Errorf("redirect to %s = %v, want ErrPrivateAddress", target.URL, err)
	}
	if got := proxy.seen(); len(got) != 1 {
		t.Errorf("proxy saw %v, want only the first hop", got)
	}
	if len(recorder.seen()) != 0 {
		t.Fatal("redirect target reached")
	}
}

func TestProxyTunnelRefusesTheProxyGivenByName(t *testing.T) {
	tunnel := &pinnedTunnel{dialer: offlineDialer(), proxyAddr: "proxy.example:3128"}
	stubLookup(t, map[string][]netip.Addr{
		"proxy.example":  {netip.MustParseAddr("8.8.4.4"), netip.MustParseAddr("2001:4860:4860::8844")},
		"alias.example":  {netip.MustParseAddr("2001:4860:4860::8844")},
		"second.example": {netip.MustParseAddr("8.8.8.8"), netip.MustParseAddr("8.8.4.4")},
		"PROXY.Example.": {netip.MustParseAddr("8.8.8.8")},
		"proxy.example.": {netip.MustParseAddr("8.8.8.8")},
	}, nil)
	for _, addr := range []string{
		"8.8.4.4:3128", "[::ffff:8.8.4.4]:3128", "[2001:4860:4860::8844]:3128",
		"alias.example:3128", "second.example:3128", "PROXY.Example.:3128", "proxy.example.:3128",
	} {
		if _, err := tunnel.dial(t.Context(), "tcp", addr); !errors.Is(err, ErrPrivateAddress) {
			t.Errorf("dial %s = %v, want ErrPrivateAddress", addr, err)
		}
	}
	mapped := &pinnedTunnel{dialer: offlineDialer(), proxyAddr: "[::ffff:8.8.4.4]:3128"}
	if _, err := mapped.dial(t.Context(), "tcp", "8.8.4.4:3128"); !errors.Is(err, ErrPrivateAddress) {
		t.Errorf("dial 8.8.4.4:3128 via proxy [::ffff:8.8.4.4]:3128 = %v, want ErrPrivateAddress", err)
	}
}

func TestProxyTunnelRefusesTheProxyAsTarget(t *testing.T) {
	tunnel := &pinnedTunnel{dialer: offlineDialer(), proxyAddr: "8.8.8.8:3128"}
	for _, addr := range []string{"8.8.8.8:3128", "[::ffff:8.8.8.8]:3128"} {
		if _, err := tunnel.dial(t.Context(), "tcp", addr); !errors.Is(err, ErrPrivateAddress) {
			t.Errorf("dial %s = %v, want ErrPrivateAddress", addr, err)
		}
	}
	stubLookup(t, map[string][]netip.Addr{"proxy.example": {netip.MustParseAddr("8.8.8.8")}}, nil)
	if _, err := tunnel.dial(t.Context(), "tcp", "proxy.example:3128"); !errors.Is(err, ErrPrivateAddress) {
		t.Errorf("dial to a name for the proxy = %v, want ErrPrivateAddress", err)
	}
}

func TestGuardedDialerRefusesEveryDialedPrivateAddress(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			_ = conn.Close()
		}
	}()
	port := strconv.Itoa(listener.Addr().(*net.TCPAddr).Port)
	guarded := guardedDialer(&net.Dialer{})
	for _, addr := range []string{"127.0.0.1:" + port, "localhost:" + port} {
		conn, err := guarded.DialContext(t.Context(), "tcp", addr)
		if conn != nil {
			_ = conn.Close()
		}
		if !errors.Is(err, ErrPrivateAddress) {
			t.Errorf("guarded dial %s = %v, want ErrPrivateAddress", addr, err)
		}
	}
	if guarded.FallbackDelay != fallbackDelay {
		t.Errorf("FallbackDelay = %v, want %v", guarded.FallbackDelay, fallbackDelay)
	}
}

func TestInterleaveAlternatesFamilies(t *testing.T) {
	in := []netip.Addr{
		netip.MustParseAddr("2606:4700::1"), netip.MustParseAddr("2606:4700::2"),
		netip.MustParseAddr("1.1.1.1"), netip.MustParseAddr("1.0.0.1"),
	}
	want := []string{"2606:4700::1", "1.1.1.1", "2606:4700::2", "1.0.0.1"}
	got := interleave(in)
	for i := range want {
		if got[i].String() != want[i] {
			t.Fatalf("interleave = %v, want %v", got, want)
		}
	}
}

func TestRaceStartsTheNextAddressAfterTheStagger(t *testing.T) {
	broken := netip.MustParseAddr("2606:4700::1")
	working := netip.MustParseAddr("1.1.1.1")
	var cancelled atomic.Bool
	began := time.Now()
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	conn, err := race(ctx, []netip.Addr{broken, working}, func(ctx context.Context, addr netip.Addr) (net.Conn, error) {
		if addr == broken {
			<-ctx.Done()
			cancelled.Store(true)
			return nil, ctx.Err()
		}
		client, server := net.Pipe()
		_ = server.Close()
		return client, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = conn.Close()
	elapsed := time.Since(began)
	if elapsed < fallbackDelay || elapsed > 2*time.Second {
		t.Errorf("race took %v, want about %v", elapsed, fallbackDelay)
	}
	time.Sleep(50 * time.Millisecond)
	if !cancelled.Load() {
		t.Error("the losing attempt was not cancelled")
	}
}

func TestRaceTriesTheNextAddressAtOnceAfterAFailure(t *testing.T) {
	var tried []string
	var mu sync.Mutex
	began := time.Now()
	_, err := race(t.Context(), []netip.Addr{netip.MustParseAddr("2606:4700::1"), netip.MustParseAddr("2606:4700::2"), netip.MustParseAddr("1.1.1.1")},
		func(_ context.Context, addr netip.Addr) (net.Conn, error) {
			mu.Lock()
			tried = append(tried, addr.String())
			mu.Unlock()
			return nil, errors.New("refused")
		})
	if err == nil || strings.Join(tried, " ") != "2606:4700::1 1.1.1.1 2606:4700::2" {
		t.Fatalf("race = %v after trying %v, want an error after all three, families interleaved", err, tried)
	}
	if time.Since(began) >= fallbackDelay {
		t.Errorf("a failed attempt waited for the stagger")
	}
}

func TestProxyTunnelFallsBackToTheOtherFamily(t *testing.T) {
	recorder := &hostRecorder{}
	backend := httptest.NewServer(recorder)
	t.Cleanup(backend.Close)
	ipv6Broken := func(target string) bool { return strings.HasPrefix(target, "[") }
	proxy := newConnectProxy(t, backend.Listener.Addr().String(), ipv6Broken)
	stubLookup(t, map[string][]netip.Addr{"dual.example": {
		netip.MustParseAddr("2606:4700::1"), netip.MustParseAddr("8.8.8.8"),
	}}, nil)
	if err := fetch(blockingClient(t, proxy.URL), "http://dual.example/v1/models"); err != nil {
		t.Fatalf("GET with a broken IPv6 path = %v, want the IPv4 fallback", err)
	}
	if got := proxy.seen(); len(got) != 2 || got[0] != "CONNECT [2606:4700::1]:80" || got[1] != "CONNECT 8.8.8.8:80" {
		t.Errorf("proxy saw %v, want IPv6 then IPv4", got)
	}
	if got := recorder.seen(); len(got) != 1 || got[0] != "dual.example" {
		t.Errorf("backend saw Host %v, want dual.example", got)
	}
}

// offlineDialer fails every name lookup and connection at once, so a test
// that should have been refused earlier fails fast instead of reaching the
// network.
func offlineDialer() *net.Dialer {
	return &net.Dialer{
		Resolver: &net.Resolver{PreferGo: true, Dial: func(context.Context, string, string) (net.Conn, error) {
			return nil, errors.New("offline test dialer")
		}},
		Control: func(string, string, syscall.RawConn) error { return errors.New("offline test dialer") },
	}
}

func TestRaceStartsTheNextAddressAtOnceWhenOneFailsWhileOthersRun(t *testing.T) {
	hanging := netip.MustParseAddr("2606:4700::1")
	failing := netip.MustParseAddr("1.1.1.1")
	working := netip.MustParseAddr("2606:4700::2")
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	began := time.Now()
	conn, err := race(ctx, []netip.Addr{hanging, working, failing}, func(ctx context.Context, addr netip.Addr) (net.Conn, error) {
		switch addr {
		case hanging:
			<-ctx.Done()
			return nil, ctx.Err()
		case failing:
			return nil, errors.New("refused")
		}
		client, server := net.Pipe()
		_ = server.Close()
		return client, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = conn.Close()
	if elapsed := time.Since(began); elapsed >= 2*fallbackDelay-50*time.Millisecond {
		t.Errorf("race took %v, want the third attempt right after the second failed (about %v)", elapsed, fallbackDelay)
	}
}
