from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote, unquote
import ipaddress

TRACKING_PARAMS = {'utm_source','utm_medium','utm_campaign','utm_term','utm_content','fbclid','gclid','msclkid','mc_cid','mc_eid'}


def normalize_url(url: str) -> dict:
    """Normalize a URL and return structured info.

    Returns dict with keys: original_url, normalized_url, scheme, netloc, path, query, is_ip, is_punycode
    """
    if not isinstance(url, str):
        url = str(url)

    original = url.strip()
    if original == '':
        return {'original_url': original, 'normalized_url': '', 'scheme': '', 'netloc': '', 'path': '', 'query': '', 'is_ip': False, 'is_punycode': False}

    # Ensure scheme
    if '://' not in original:
        original = 'https://' + original

    parsed = urlparse(original)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc

    # Remove default ports
    if netloc.endswith(':80') and scheme == 'http':
        netloc = netloc.rsplit(':', 1)[0]
    if netloc.endswith(':443') and scheme == 'https':
        netloc = netloc.rsplit(':', 1)[0]

    import re

    # Detect punycode and decode where possible
    is_punycode = 'xn--' in netloc
    try:
        # split port if present
        host = netloc.split(':')[0]
        hostname = host.encode('utf-8').decode('idna') if is_punycode else host
    except Exception:
        hostname = host

    fixes = []
    # Heuristic correction: common leading 'w' typos like 'ww.' -> 'www.'
    m = re.match(r'^(w{2,})\.(.+)$', hostname, flags=re.I)
    if m and m.group(1).lower() != 'www':
        corrected = 'www.' + m.group(2)
        fixes.append(f"Corrected leading 'w' sequence from '{hostname}' to '{corrected}'")
        hostname = corrected

    # Detect IP
    is_ip = False
    try:
        ipaddress.ip_address(hostname)
        is_ip = True
    except Exception:
        is_ip = False

    # Normalize path: unquote then quote to remove weird encodings
    path = unquote(parsed.path or '/')
    path = quote(path, safe='/%')

    # Filter tracking query params and sort the rest
    qsl = parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [(k, v) for (k, v) in qsl if k not in TRACKING_PARAMS]
    filtered.sort()
    query = urlencode(filtered, doseq=True)

    # Rebuild normalized url with possible corrected host
    # Preserve port if present
    port = ''
    if ':' in netloc:
        parts = netloc.rsplit(':', 1)
        if len(parts) == 2 and parts[1].isdigit():
            port = ':' + parts[1]

    netloc_corrected = hostname.lower() + port
    normalized = urlunparse((scheme, netloc_corrected, path.rstrip('/'), '', query, ''))

    return {
        'original_url': url,
        'normalized_url': normalized,
        'scheme': scheme,
        'netloc': netloc_corrected,
        'path': path,
        'query': query,
        'is_ip': is_ip,
        'is_punycode': is_punycode,
        'fixes': fixes
    }
