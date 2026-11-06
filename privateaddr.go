package llm

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"sync"
	"syscall"

	"github.com/bds421/rho-kit/security/v2/netutil"
)

// ErrPrivateAddress is returned, wrapped, when Config.BlockPrivateBaseURL
// refuses a connection to a private, reserved or otherwise non-public address.
// It is not retryable.
var ErrPrivateAddress = errors.New("llm: private or reserved address refused (BlockPrivateBaseURL)")

// wellKnownNAT64 is 64:ff9b::/96 (RFC 6052). An address in it is judged by the
// IPv4 address it carries, so an IPv6-only host behind DNS64 still reaches
// public providers while 64:ff9b::a9fe:a9fe (169.254.169.254) stays refused.
var wellKnownNAT64 = func() netutil.NAT64Prefix {
	prefix, err := netutil.ParseNAT64Prefix("64:ff9b::/96")
	if err != nil {
		panic("llm: invalid well-known NAT64 prefix")
	}
	return prefix
}()

// privateAddr reports whether BlockPrivateBaseURL refuses addr: rho-kit's
// netutil.IsPrivateIP, which also covers IPv4-mapped, IPv4-compatible, 6to4,
// Teredo and other IPv6 forms that carry an IPv4 address.
func privateAddr(addr netip.Addr) bool {
	addr = addr.WithZone("")
	if v4, ok := wellKnownNAT64.Extract(addr); ok {
		addr = v4
	}
	return netutil.IsPrivateIP(net.IP(addr.AsSlice()))
}

// blockPrivateDialContext refuses, at connect time, every address privateAddr
// refuses. It covers what CheckBaseURL cannot: host names (checked after
// resolution, on the address actually dialed, so DNS rebinding does not get
// through) and redirects. The configured or ambient proxy is a reviewed
// egress point and may be private; proxies records the addresses it dials.
func blockPrivateDialContext(dialer *net.Dialer, proxies *proxyAddrs) func(context.Context, string, string) (net.Conn, error) {
	guarded := *dialer
	guarded.Control = func(_, address string, _ syscall.RawConn) error {
		target, err := netip.ParseAddrPort(address)
		if err != nil || privateAddr(target.Addr()) {
			return fmt.Errorf("%w: %s", ErrPrivateAddress, address)
		}
		return nil
	}
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		if proxies.has(addr) {
			return dialer.DialContext(ctx, network, addr)
		}
		return guarded.DialContext(ctx, network, addr)
	}
}

// proxyAddrs remembers the host:port of every proxy the transport's proxy
// policy has chosen, so the dial guard can tell a proxy connection apart.
type proxyAddrs struct {
	seen sync.Map
}

func (p *proxyAddrs) record(policy func(*http.Request) (*url.URL, error)) func(*http.Request) (*url.URL, error) {
	if policy == nil {
		return nil
	}
	return func(req *http.Request) (*url.URL, error) {
		proxyURL, err := policy(req)
		if proxyURL != nil {
			p.seen.Store(proxyDialAddr(proxyURL), struct{}{})
		}
		return proxyURL, err
	}
}

func (p *proxyAddrs) has(addr string) bool {
	_, ok := p.seen.Load(addr)
	return ok
}

// proxyDialAddr is the address net/http dials for a proxy URL.
func proxyDialAddr(u *url.URL) string {
	port := u.Port()
	if port == "" {
		port = map[string]string{"http": "80", "https": "443", "socks5": "1080", "socks5h": "1080"}[u.Scheme]
	}
	return net.JoinHostPort(u.Hostname(), port)
}
