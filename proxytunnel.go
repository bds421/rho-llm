package llm

import (
	"bufio"
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"
)

// pinnedTunnel reaches targets through an explicit proxy with CONNECT to an
// address rho-llm resolved and checked itself (see blockPrivateEgress).
type pinnedTunnel struct {
	dialer    *net.Dialer
	proxy     *url.URL
	proxyAddr string
}

func (p *pinnedTunnel) dial(ctx context.Context, network, addr string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrPrivateAddress, addr)
	}
	addrs, err := resolvePublic(ctx, host)
	if err != nil {
		return nil, err
	}
	if err := p.refuseSelf(ctx, host, port, addrs); err != nil {
		return nil, err
	}
	return race(ctx, addrs, func(ctx context.Context, target netip.Addr) (net.Conn, error) {
		return p.open(ctx, network, net.JoinHostPort(target.String(), port))
	})
}

// open connects to the proxy and opens a CONNECT tunnel to target.
func (p *pinnedTunnel) open(ctx context.Context, network, target string) (net.Conn, error) {
	conn, err := p.dialer.DialContext(ctx, network, p.proxyAddr)
	if err != nil {
		return nil, err
	}
	if p.proxy.Scheme == "https" {
		tlsConn := tls.Client(conn, &tls.Config{ServerName: p.proxy.Hostname(), MinVersion: tls.VersionTLS12})
		if err := tlsConn.HandshakeContext(ctx); err != nil {
			_ = conn.Close()
			return nil, &net.OpError{Op: "proxyconnect", Net: network, Err: err}
		}
		conn = tlsConn
	}
	if err := connect(ctx, conn, target); err != nil {
		_ = conn.Close()
		return nil, &net.OpError{Op: "proxyconnect", Net: network, Err: err}
	}
	return conn, nil
}

// refuseSelf refuses a target that is the proxy itself: the same normalized
// name, or a checked address the proxy host resolves to, on the proxy port.
func (p *pinnedTunnel) refuseSelf(ctx context.Context, host, port string, checked []netip.Addr) error {
	proxyHost, proxyPort, err := net.SplitHostPort(p.proxyAddr)
	if err != nil || port != proxyPort {
		return err
	}
	if normalizeHost(host) == normalizeHost(proxyHost) {
		return fmt.Errorf("%w: %s is the proxy", ErrPrivateAddress, host)
	}
	proxyAddrs := []netip.Addr{}
	if literal, err := netip.ParseAddr(proxyHost); err == nil {
		proxyAddrs = append(proxyAddrs, literal)
	} else if proxyAddrs, err = lookupNetIP(ctx, "ip", proxyHost); err != nil {
		return fmt.Errorf("llm: resolve proxy %s: %w", proxyHost, err)
	}
	for _, a := range proxyAddrs {
		for _, c := range checked {
			if a.Unmap().WithZone("") == c.Unmap().WithZone("") {
				return fmt.Errorf("%w: %s resolves to the proxy %s", ErrPrivateAddress, host, a)
			}
		}
	}
	return nil
}

// normalizeHost lower-cases a host name and drops a trailing dot.
func normalizeHost(host string) string {
	return strings.TrimSuffix(strings.ToLower(host), ".")
}

// connect opens an HTTP CONNECT tunnel to target over conn.
func connect(ctx context.Context, conn net.Conn, target string) error {
	stop := context.AfterFunc(ctx, func() { _ = conn.SetDeadline(time.Unix(1, 0)) })
	err := writeConnect(conn, target)
	if !stop() {
		return ctx.Err()
	}
	if err != nil {
		return err
	}
	return conn.SetDeadline(time.Time{})
}

func writeConnect(conn net.Conn, target string) error {
	req := &http.Request{
		Method: http.MethodConnect,
		URL:    &url.URL{Opaque: target},
		Host:   target,
		Header: http.Header{},
	}
	if err := req.Write(conn); err != nil {
		return err
	}
	reader := bufio.NewReader(conn)
	resp, err := http.ReadResponse(reader, req)
	if err != nil {
		return err
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("llm: proxy refused CONNECT %s: %s", target, resp.Status)
	}
	if reader.Buffered() > 0 {
		return fmt.Errorf("llm: proxy sent data before the tunnel was used")
	}
	return nil
}

// dialAddr is the host:port net/http dials for u.
func dialAddr(u *url.URL) string {
	port := u.Port()
	if port == "" {
		port = map[string]string{"http": "80", "https": "443"}[u.Scheme]
	}
	return net.JoinHostPort(normalizeHost(u.Hostname()), port)
}
