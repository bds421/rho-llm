package llm

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"strings"
	"syscall"
	"time"
)

// lookupNetIP resolves target names for the proxy tunnel, which connects only
// to the checked addresses.
var lookupNetIP = net.DefaultResolver.LookupNetIP

// dialResolver resolves names for the direct dialer; nil is the system's.
// Tests answer from it, so a name resolves alike on every platform.
var dialResolver *net.Resolver

// fallbackDelay is how long a connection attempt runs alone before the next
// checked address starts racing it (the Happy Eyeballs stagger, RFC 8305).
const fallbackDelay = 300 * time.Millisecond

type dialFunc func(ctx context.Context, network, addr string) (net.Conn, error)

// blockPrivateEgress returns the dial function of a client with
// BlockPrivateBaseURL set. Ambient proxy variables are ignored; the transport
// gets no Proxy, so every connection, each redirect hop included, goes
// through this function.
//
// Without ProxyURL the dialer resolves and races IPv6 and IPv4 as usual and
// refuses, just before connecting, every address that is private or reserved;
// each address actually dialed is checked, so DNS rebinding cannot get
// through.
//
// With ProxyURL it resolves and checks the target itself and opens a CONNECT
// tunnel through the proxy to a checked IP address, racing the addresses with
// the same stagger, so the proxy cannot resolve the name to another address.
// TLS then runs inside the tunnel with the original name for SNI and
// certificate checks, and plain HTTP sends it as Host. A proxy that filters
// by domain name sees only IP addresses.
func blockPrivateEgress(dialer *net.Dialer, proxyURL string) (dialFunc, error) {
	proxyURL = strings.TrimSpace(proxyURL)
	if proxyURL == "" {
		return guardedDialer(dialer).DialContext, nil
	}
	proxy, err := parseProxyURL(proxyURL)
	if err != nil {
		return nil, err
	}
	tunnel := &pinnedTunnel{dialer: dialer, proxy: proxy, proxyAddr: dialAddr(proxy)}
	return tunnel.dial, nil
}

// guardedDialer is dialer refusing, just before each connect, an address
// that is private or reserved.
func guardedDialer(dialer *net.Dialer) *net.Dialer {
	guarded := *dialer
	guarded.FallbackDelay = fallbackDelay
	guarded.ControlContext = func(_ context.Context, _, address string, _ syscall.RawConn) error {
		target, err := netip.ParseAddrPort(address)
		if err != nil || privateAddr(target.Addr()) {
			return fmt.Errorf("%w: %s", ErrPrivateAddress, address)
		}
		return nil
	}
	return &guarded
}

// resolvePublic returns the checked addresses for host. A name is refused when
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

// interleave alternates the address families, keeping the order within each
// and starting with the family of the first address (RFC 8305).
func interleave(addrs []netip.Addr) []netip.Addr {
	if len(addrs) == 0 {
		return addrs
	}
	var first, second []netip.Addr
	for _, a := range addrs {
		if a.Is4() == addrs[0].Is4() {
			first = append(first, a)
		} else {
			second = append(second, a)
		}
	}
	out := make([]netip.Addr, 0, len(addrs))
	for i := 0; i < len(first) || i < len(second); i++ {
		if i < len(first) {
			out = append(out, first[i])
		}
		if i < len(second) {
			out = append(out, second[i])
		}
	}
	return out
}

// race runs attempt for each address in interleaved order, starting the next
// one after fallbackDelay or as soon as any running attempt fails (RFC 8305). The first
// connection wins; the others are cancelled and closed.
func race(ctx context.Context, addrs []netip.Addr, attempt func(context.Context, netip.Addr) (net.Conn, error)) (net.Conn, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	type result struct {
		conn net.Conn
		err  error
	}
	order := interleave(addrs)
	results := make(chan result, len(order))
	started, pending := 0, 0
	start := func() {
		addr := order[started]
		started++
		pending++
		go func() {
			conn, err := attempt(ctx, addr)
			results <- result{conn, err}
		}()
	}
	start()
	timer := time.NewTimer(fallbackDelay)
	defer timer.Stop()
	var errs []error
	for pending > 0 {
		select {
		case r := <-results:
			pending--
			if r.err == nil {
				go func(n int) {
					for range n {
						if late := <-results; late.conn != nil {
							_ = late.conn.Close()
						}
					}
				}(pending)
				return r.conn, nil
			}
			errs = append(errs, r.err)
			if started < len(order) {
				start()
				timer.Reset(fallbackDelay)
			}
		case <-timer.C:
			if started < len(order) {
				start()
				timer.Reset(fallbackDelay)
			}
		}
	}
	return nil, errors.Join(errs...)
}
