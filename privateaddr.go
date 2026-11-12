package llm

import (
	"errors"
	"fmt"
	"net/netip"
	"strings"
)

// ErrPrivateAddress is returned, wrapped, when Config.BlockPrivateBaseURL
// refuses a host or a connection to a private, reserved or otherwise
// non-public address. It is not retryable.
var ErrPrivateAddress = errors.New("llm: private or reserved address refused (BlockPrivateBaseURL)")

// nonPublicPrefixes are the ranges BlockPrivateBaseURL refuses. IPv4-mapped
// addresses are unmapped first, and an address in the NAT64 well-known prefix
// is judged by the IPv4 address it carries (see privateAddr).
var nonPublicPrefixes = func() []netip.Prefix {
	cidrs := []string{
		"0.0.0.0/8",        // "this network" (RFC 791)
		"10.0.0.0/8",       // private (RFC 1918)
		"100.64.0.0/10",    // shared address space, CGNAT (RFC 6598)
		"127.0.0.0/8",      // loopback
		"169.254.0.0/16",   // link-local, cloud metadata
		"168.63.129.16/32", // Azure wire server: platform metadata, DHCP, health probes (B36)
		"172.16.0.0/12",    // private (RFC 1918)
		"192.0.0.0/24",     // IETF protocol assignments (RFC 6890)
		"192.0.2.0/24",     // documentation TEST-NET-1 (RFC 5737)
		"192.88.99.0/24",   // 6to4 relay anycast (RFC 7526)
		"192.168.0.0/16",   // private (RFC 1918)
		"198.18.0.0/15",    // benchmarking (RFC 2544)
		"198.51.100.0/24",  // documentation TEST-NET-2
		"203.0.113.0/24",   // documentation TEST-NET-3
		"224.0.0.0/4",      // multicast
		"240.0.0.0/4",      // reserved, broadcast
		"::/96",            // unspecified, loopback, IPv4-compatible (RFC 4291)
		"::ffff:0:0:0/96",  // IPv4-translated (RFC 2765)
		"64:ff9b:1::/48",   // local-use IPv4/IPv6 translation (RFC 8215)
		"100::/64",         // discard-only (RFC 6666)
		"2001::/32",        // Teredo, carries an IPv4 address
		"2001:2::/48",      // benchmarking (RFC 5180)
		"2001:10::/28",     // ORCHID (RFC 4843)
		"2001:db8::/32",    // documentation (RFC 3849)
		"2002::/16",        // 6to4, carries an IPv4 address
		"3fff::/20",        // documentation (RFC 9637)
		"5f00::/16",        // SRv6 SIDs (RFC 9602)
		"fc00::/7",         // unique local
		"fe80::/10",        // link-local
		"fec0::/10",        // site-local, deprecated
		"ff00::/8",         // multicast
	}
	prefixes := make([]netip.Prefix, len(cidrs))
	for i, cidr := range cidrs {
		prefixes[i] = netip.MustParsePrefix(cidr)
	}
	return prefixes
}()

// wellKnownNAT64 is 64:ff9b::/96 (RFC 6052). An address in it is judged by the
// IPv4 address it carries, so an IPv6-only host behind DNS64 still reaches
// public providers while 64:ff9b::a9fe:a9fe (169.254.169.254) stays refused.
var wellKnownNAT64 = netip.MustParsePrefix("64:ff9b::/96")

// privateAddr reports whether BlockPrivateBaseURL refuses addr.
func privateAddr(addr netip.Addr) bool {
	addr = addr.WithZone("").Unmap()
	if wellKnownNAT64.Contains(addr) {
		b := addr.As16()
		addr = netip.AddrFrom4([4]byte{b[12], b[13], b[14], b[15]})
	}
	if !addr.IsValid() || addr.IsUnspecified() || addr.IsLoopback() || addr.IsPrivate() ||
		addr.IsLinkLocalUnicast() || addr.IsLinkLocalMulticast() || addr.IsMulticast() {
		return true
	}
	for _, prefix := range nonPublicPrefixes {
		if prefix.Contains(addr) {
			return true
		}
	}
	return false
}

// checkHost refuses a host BlockPrivateBaseURL never lets through without
// resolving it: "localhost" and its subdomains, a private IP literal, and a
// host a URL parser or resolver may read as a non-canonical IPv4 address
// (0x7f.1, 2130706433, 0177.0.0.1). It reports whether host is an IP literal.
func checkHost(host string) (netip.Addr, bool, error) {
	host = strings.TrimSuffix(host, ".")
	if addr, err := netip.ParseAddr(host); err == nil {
		if privateAddr(addr) {
			return addr, true, fmt.Errorf("%w: %s", ErrPrivateAddress, host)
		}
		return addr, true, nil
	}
	lower := strings.ToLower(host)
	if lower == "" || lower == "localhost" || strings.HasSuffix(lower, ".localhost") {
		return netip.Addr{}, false, fmt.Errorf("%w: %q", ErrPrivateAddress, host)
	}
	if numericHost(lower) {
		return netip.Addr{}, false, fmt.Errorf("%w: %q looks like an IPv4 address", ErrPrivateAddress, host)
	}
	return netip.Addr{}, false, nil
}

// numericHost applies the WHATWG URL rule "ends in a number" (last label all
// digits, or 0x followed by hex digits) and also refuses any host starting
// with 0x.
func numericHost(host string) bool {
	if strings.HasPrefix(host, "0x") {
		return true
	}
	last := host[strings.LastIndex(host, ".")+1:]
	if hex, ok := strings.CutPrefix(last, "0x"); ok {
		return strings.Trim(hex, "0123456789abcdef") == ""
	}
	return last != "" && strings.Trim(last, "0123456789") == ""
}
