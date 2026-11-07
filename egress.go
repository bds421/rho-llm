package llm

import (
	"bufio"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"
)

// lookupNetIP resolves target names; the dial then uses only the checked
// addresses.
var lookupNetIP = net.DefaultResolver.LookupNetIP

// fallbackDialTimeout bounds a connection attempt that has another address
// to fall back to.
const fallbackDialTimeout = 5 * time.Second

type dialFunc func(ctx context.Context, network, addr string) (net.Conn, error)

// blockPrivateEgress returns the dial function of a client with
// BlockPrivateBaseURL set. Ambient proxy variables are ignored; the transport
// gets no Proxy, so every connection, each redirect hop included, goes
// through this function.
//
// Without ProxyURL it resolves the target itself, refuses the host when the
// lookup fails, returns nothing or returns any private or reserved address,
// and dials only the checked addresses, so DNS rebinding cannot swap them.
//
// With ProxyURL it resolves and checks the target itself and opens a CONNECT
// tunnel through the proxy to the checked IP address, so the proxy cannot
// resolve the name to another address. TLS then runs inside the tunnel with
// the original name for SNI and certificate checks, and plain HTTP sends it
// as Host. A proxy that filters by domain name sees only the IP address.
func blockPrivateEgress(dialer *net.Dialer, proxyURL string) (dialFunc, error) {
	proxyURL = strings.TrimSpace(proxyURL)
	if proxyURL == "" {
		return func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, fmt.Errorf("%w: %s", ErrPrivateAddress, addr)
			}
			addrs, err := resolvePublic(ctx, host)
			if err != nil {
				return nil, err
			}
			return dialInTurn(ctx, dialer, network, addrs, port)
		}, nil
	}
	proxy, err := parseProxyURL(proxyURL)
	if err != nil {
		return nil, err
	}
	tunnel := &pinnedTunnel{dialer: dialer, proxy: proxy, proxyAddr: dialAddr(proxy)}
	return tunnel.dial, nil
}

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
	target := net.JoinHostPort(addrs[0].String(), port)
	if strings.EqualFold(net.JoinHostPort(host, port), p.proxyAddr) || target == p.proxyAddr {
		return nil, fmt.Errorf("%w: %s is the proxy", ErrPrivateAddress, addr)
	}
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

// resolvePublic returns the address to pin for host. A name is refused when
// its lookup fails, returns nothing, or returns any private address.
func resolvePublic(ctx context.Context, host string) ([]netip.Addr, error) {
	addr, literal, err := checkHost(host)
	if err != nil {
		return nil, err
	}
	if literal {
		return []netip.Addr{addr.Unmap()}, nil
	}
	addrs, err := lookupNetIP(ctx, "ip", host)
	if err != nil {
		return nil, fmt.Errorf("%w: resolve %s: %v", ErrPrivateAddress, host, err)
	}
	if len(addrs) == 0 {
		return nil, fmt.Errorf("%w: %s resolves to no address", ErrPrivateAddress, host)
	}
	for _, a := range addrs {
		if privateAddr(a) {
			return nil, fmt.Errorf("%w: %s resolves to %s", ErrPrivateAddress, host, a)
		}
	}
	public := make([]netip.Addr, len(addrs))
	for i, a := range addrs {
		public[i] = a.Unmap()
	}
	return public, nil
}

// dialInTurn dials the checked addresses one after the other, so a broken
// IPv6 or IPv4 path falls back to the other family. Every address but the
// last gets at most fallbackDialTimeout.
func dialInTurn(ctx context.Context, dialer *net.Dialer, network string, addrs []netip.Addr, port string) (net.Conn, error) {
	var errs []error
	for i, addr := range addrs {
		attempt := *dialer
		if i < len(addrs)-1 && (attempt.Timeout == 0 || attempt.Timeout > fallbackDialTimeout) {
			attempt.Timeout = fallbackDialTimeout
		}
		conn, err := attempt.DialContext(ctx, network, net.JoinHostPort(addr.String(), port))
		if err == nil {
			return conn, nil
		}
		errs = append(errs, err)
		if ctx.Err() != nil {
			break
		}
	}
	return nil, errors.Join(errs...)
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
	return net.JoinHostPort(strings.ToLower(u.Hostname()), port)
}
