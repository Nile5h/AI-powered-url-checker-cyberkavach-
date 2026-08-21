from .url_normalize import normalize_url
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parents[1]


def extract_url_features(url):
    """
    Extract richer features from a URL using a normalized form.

    Returns a fixed-length list of numeric features in the following order:
    0: normalized URL length
    1: number of dots
    2: number of hyphens
    3: number of @ symbols
    4: is HTTPS (1/0)
    5: host dot count (domain parts)
    6: path depth (number of path segments)
    7: number of digits
    8: suspicious keyword flag (1/0)
    9: underscore count
    10: query parameter count
    11: is IP (1/0)
    12: is punycode (1/0)
    """
    suspicious_keywords = ['free', 'click', 'verify', 'confirm', 'urgent', 'update', 'login', 'secure', 'bank', 'account', 'claim', 'prize', 'win']

    info = normalize_url(url)
    n_url = info.get('normalized_url', '')
    host = info.get('netloc', '')
    path = info.get('path', '')
    query = info.get('query', '')

    url_len = len(n_url)
    dot_count = n_url.count('.')
    hyphen_count = n_url.count('-')
    at_count = n_url.count('@')

    is_https = 1 if info.get('scheme') == 'https' else 0

    host_dot_count = host.count('.')
    # path depth: count non-empty segments
    path_depth = len([p for p in path.split('/') if p])

    digit_count = sum(1 for c in n_url if c.isdigit())
    has_suspicious = 1 if any(keyword in n_url.lower() for keyword in suspicious_keywords) else 0
    underscore_count = n_url.count('_')
    query_param_count = 0 if query == '' else query.count('&') + 1

    is_ip = 1 if info.get('is_ip') else 0
    is_punycode = 1 if info.get('is_punycode') else 0

    # Whitelist check (optional file: dataset/whitelist.txt)
    is_whitelisted = 0
    try:
        host = info.get('netloc','').lower()
        # load once per import
        global _WHITELIST
        if '_WHITELIST' not in globals():
            _WHITELIST = set()
            try:
                with open(DATASET_DIR / 'whitelist.txt', 'r', encoding='utf-8') as fh:
                    for line in fh:
                        _WHITELIST.add(line.strip().lower())
            except Exception:
                _WHITELIST = set()
        if host in _WHITELIST:
            is_whitelisted = 1
    except Exception:
        is_whitelisted = 0

    return [
        url_len,
        dot_count,
        hyphen_count,
        at_count,
        is_https,
        host_dot_count,
        path_depth,
        digit_count,
        has_suspicious,
        underscore_count,
        query_param_count,
        is_ip,
        is_punycode,
        is_whitelisted
    ]