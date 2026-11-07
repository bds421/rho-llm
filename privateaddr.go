package llm

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
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

// blockPrivateEgress returns the proxy policy and dialer of a client with
// BlockPrivateBaseURL set. Without an explicit ProxyURL (ambient proxy
// variables are ignored) the dialer refuses every private or reserved address
// at connect time, after resolution, so host names, redirects and DNS
// rebinding are covered. With ProxyURL, the dialer reaches only the proxy, and
// every request (each redirect hop is one) is checked before it is handed to
// the proxy: its host must not be the proxy itself, "localhost" or a private
// literal, and a name must not resolve to a private address. The proxy
// resolves the name again, so it has to enforce egress filtering itself.
func blockPrivateEgress(dialer *net.Dialer, proxyURL string) (func(*http.Request) (*url.URL, error), func(context.Context, string, string) (net.Conn, error)) {
	proxyURL = strings.TrimSpace(proxyURL)
	if proxyURL == "" {
		guarded := *dialer
		guarded.Control = func(_, address string, _ syscall.RawConn) error {
			target, err := netip.ParseAddrPort(address)
			if err != nil || privateAddr(target.Addr()) {
				return fmt.Errorf("%w: %s", ErrPrivateAddress, address)
			}
			return nil
		}
		return nil, guarded.DialContext
	}
	proxy, err := parseProxyURL(proxyURL)
	if err != nil {
		return func(*http.Request) (*url.URL, error) { return nil, err }, dialer.DialContext
	}
	proxyAddr := dialAddr(proxy)
	policy := func(req *http.Request) (*url.URL, error) {
		if err := checkProxiedTarget(req, proxyAddr); err != nil {
			return nil, err
		}
		return proxy, nil
	}
	dial := func(ctx context.Context, network, addr string) (net.Conn, error) {
		if addr != proxyAddr {
			return nil, fmt.Errorf("%w: %s is not the proxy", ErrPrivateAddress, addr)
		}
		return dialer.DialContext(ctx, network, addr)
	}
	return policy, dial
}

// checkProxiedTarget refuses a request that the proxy would send to a private
// address, or that targets the proxy itself.
func checkProxiedTarget(req *http.Request, proxyAddr string) error {
	if req == nil || req.URL == nil {
		return fmt.Errorf("%w: request without URL", ErrPrivateAddress)
	}
	target := dialAddr(req.URL)
	if target == proxyAddr {
		return fmt.Errorf("%w: %s is the proxy", ErrPrivateAddress, target)
	}
	host := strings.TrimSuffix(req.URL.Hostname(), ".")
	if strings.EqualFold(host, "localhost") || strings.HasSuffix(strings.ToLower(host), ".localhost") {
		return fmt.Errorf("%w: %s", ErrPrivateAddress, host)
	}
	if addr, err := netip.ParseAddr(host); err == nil {
		if privateAddr(addr) {
			return fmt.Errorf("%w: %s", ErrPrivateAddress, host)
		}
		return nil
	}
	addrs, err := lookupNetIP(req.Context(), "ip", host)
	if err != nil {
		return fmt.Errorf("llm: resolve %s: %w", host, err)
	}
	for _, addr := range addrs {
		if privateAddr(addr) {
			return fmt.Errorf("%w: %s resolves to %s", ErrPrivateAddress, host, addr)
		}
	}
	return nil
}

// lookupNetIP resolves names before a request is handed to the proxy.
var lookupNetIP = net.DefaultResolver.LookupNetIP

// dialAddr is the host:port net/http dials for u.
func dialAddr(u *url.URL) string {
	port := u.Port()
	if port == "" {
		port = map[string]string{"http": "80", "https": "443"}[u.Scheme]
	}
	return net.JoinHostPort(strings.ToLower(u.Hostname()), port)
}
