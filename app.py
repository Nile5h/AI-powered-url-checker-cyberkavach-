import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pickle

try:
    from dataset.utils.url_features import extract_url_features
except (ModuleNotFoundError, ImportError) as e:
    st.error(f"Error: Failed to import extract_url_features. Details: {e}")
    st.stop()

# Load models and vectorizer
try:
    text_model = pickle.load(open("model/text_model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
    # Prefer calibrated URL model if available (safer probabilities)
    if os.path.exists('model/url_model_calibrated.pkl'):
        url_model = pickle.load(open('model/url_model_calibrated.pkl', 'rb'))
        calibrated_model_used = True
    else:
        url_model = pickle.load(open("model/url_model.pkl", "rb"))
        calibrated_model_used = False
except (FileNotFoundError, pickle.UnpicklingError) as e:
    st.error(f"Error: Failed to load model files. Details: {e}")
    st.stop()

phish_lookup = {}
try:
    import pandas as pd
    phish_df = pd.read_csv('dataset/verified_online.csv')
    phish_df = phish_df[(phish_df['verified'].str.lower() == 'yes') & (phish_df['online'].str.lower() == 'yes')]
    from dataset.utils.url_normalize import normalize_url
    phish_df['normalized'] = phish_df['url'].fillna('').apply(lambda u: normalize_url(u).get('normalized_url',''))
    phish_lookup = {r['normalized']: r.to_dict() for _, r in phish_df.iterrows()}
    # st.write(f"Loaded {len(phish_lookup)} verified phishing records for lookup")
except Exception:
    phish_lookup = {}
    st.write('No verified phishing lookup available')

# --- External checks and logging helpers ---
try:
    import requests
except Exception:
    requests = None
import csv
from datetime import datetime
import time
import base64
import logging
logging.basicConfig(level=logging.INFO)
from urllib.parse import urlparse, urlunparse, quote, parse_qsl, urlencode


def save_raw_response(source, url, response_obj):
    """Save any raw JSON/text response to a timestamped file and return its path."""
    outdir = os.path.join('dataset', 'raw_responses')
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
    fname = f"{ts}_{source}.json"
    path = os.path.join(outdir, fname)
    try:
        import json as _json
        with open(path, 'w', encoding='utf-8') as fh:
            _json.dump({'url': url, 'response': response_obj}, fh, ensure_ascii=False, indent=2)
        return path
    except Exception:
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(str(response_obj))
            return path
        except Exception:
            return ''


def make_request_with_retries(method, url, timeout=10, max_retries=3, backoff_factor=0.5, proxies=None, **kwargs):
    """Make an HTTP request with retries and exponential backoff.

    Returns either a requests.Response instance on success, or a dict {'error': str(...)} on failure.
    Retries on network errors and on status codes: 429, 500, 502, 503, 504.
    """
    retry_statuses = {429, 500, 502, 503, 504}
    attempt = 0
    last_exc = None
    while True:
        try:
            if method.lower() == 'get':
                r = requests.get(url, timeout=timeout, proxies=proxies, **kwargs)
            else:
                r = requests.post(url, timeout=timeout, proxies=proxies, **kwargs)
            # If server returned a valuable error we may still want to log it for debugging
            if r.status_code in retry_statuses and attempt < max_retries:
                # save debug info for failed attempt
                try:
                    save_raw_response('request_retry', url, {'attempt': attempt+1, 'status': r.status_code, 'text': r.text, 'headers': dict(r.headers)})
                except Exception:
                    pass
                sleep = backoff_factor * (2 ** attempt)
                time.sleep(sleep)
                attempt += 1
                continue
            return r
        except Exception as exc:
            last_exc = exc
            # log network error attempts
            try:
                save_raw_response('request_error', url, {'attempt': attempt+1, 'error': str(exc)})
            except Exception:
                pass
            if attempt < max_retries:
                sleep = backoff_factor * (2 ** attempt)
                time.sleep(sleep)
                attempt += 1
                continue
            return {'error': str(last_exc)}


# --- URL sanitization helper ---
def sanitize_url(url: str) -> str:
    """Sanitize and normalize a URL for external API calls.

    - strips whitespace/newlines
    - parses and normalizes via urllib.parse
    - percent-encodes path and query safely
    - removes trailing slashes
    """
    if not url:
        return url
    u = url.strip()
    try:
        p = urlparse(u, scheme='http')
        scheme = p.scheme if p.scheme else 'http'
        if p.netloc:
            netloc = p.netloc
            path = p.path or ''
        else:
            # p.path might be 'example.com' or 'example.com/path'
            if p.path and '/' in p.path:
                netloc, _, rest = p.path.partition('/')
                netloc = netloc
                path = '/' + rest
            else:
                netloc = p.path or ''
                path = ''
        # percent-encode path, preserving '/'
        path_enc = quote(path, safe='/')
        qsl = parse_qsl(p.query, keep_blank_values=True)
        query_enc = urlencode(qsl, doseq=True)
        if path_enc.endswith('/'):
            path_enc = path_enc.rstrip('/')
        cleaned = urlunparse((scheme, netloc, path_enc, p.params or '', query_enc or '', p.fragment or ''))
        return cleaned
    except Exception:
        return u


# --- Model helper: prepare URL for model prediction ---
from dataset.utils.url_rules import SUSPICIOUS_KEYWORDS

def prepare_url_for_model(url: str) -> str:
    """Prepare a URL string for feature extraction and model prediction.

    - remove protocol (http/https///)
    - convert to lowercase
    - strip trailing slashes
    - return netloc+path[+?query] as a compact string
    """
    if not url:
        return url
    try:
        # Use sanitize_url so we have a cleaned form first
        cleaned = sanitize_url(url)
        p = urlparse(cleaned)
        netloc = p.netloc.lower()
        path = (p.path or '').rstrip('/')
        query = p.query or ''
        model_str = netloc
        if path:
            model_str += path
        if query:
            model_str += '?' + query
        return model_str
    except Exception:
        return url.strip().lower().rstrip('/')


def compute_rule_risk(normalized_info: dict, rule_reasons: list) -> tuple:
    """Compute a rule-based risk penalty and whether risk is considered 'high'.

    Returns (penalty_float_between_0_and_1, high_risk_bool, details_list)
    """
    penalty = 0.0
    details = []
    high_risk = False

    netloc = (normalized_info.get('netloc') or '').lower()
    path = (normalized_info.get('path') or '')
    url_text = (netloc + ' ' + path).lower()

    # Suspicious keywords
    found_kw = [k for k in SUSPICIOUS_KEYWORDS if k in url_text]
    if found_kw:
        penalty += 0.20
        details.append(f"suspicious_keywords:{','.join(found_kw)}")

    # IP address host
    if normalized_info.get('is_ip'):
        penalty += 0.30
        details.append('host_is_ip')

    hyphens = netloc.count('-')
    if hyphens > 3:
        penalty += 0.10
        details.append('excessive_hyphens')

    sub_depth = len(netloc.split('.')) - 1
    if sub_depth > 3:
        penalty += 0.10
        details.append('deep_subdomain')

    import re
    if re.search(r'\d{4}', netloc) or re.search(r'(20\d{2})', netloc):
        penalty += 0.15
        details.append('recent_registration_heuristic')

    # Consider domain validity and hard rule reasons as high risk
    hard_reasons = [r for r in (rule_reasons or []) if 'Not HTTPS' not in r]
    if (not normalized_info.get('is_ip') and not normalized_info.get('netloc')) or len(hard_reasons) > 0:
        high_risk = True

    # Cap penalty
    if penalty > 0.6:
        penalty = 0.6

    # If major hard reasons present mark high risk
    if any(r.startswith('Contains') or r.startswith('Host is') or r.startswith('Token-level anomalies') for r in hard_reasons):
        high_risk = True

    # Also high risk if domain_valid is False
    if normalized_info and 'netloc' in normalized_info:
        host = normalized_info.get('netloc','')
        if not host:
            high_risk = True

    return penalty, high_risk, details


def query_virustotal(url, api_key, timeout=10):
    """Query VirusTotal for a URL and return a short summary dict and raw response.

    - Uses sanitized URL values (sanitize_url) for submission
    - POSTs form data: data={'url': clean_url}
    - If POST returns 400 (canonicalization error) falls back to GET /api/v3/urls/{url_id}
    - Adds debug logging and saves diagnostics to files
    """
    if not api_key:
        return {'ok': False, 'error': 'no_api_key'}
    if requests is None:
        return {'ok': False, 'error': 'requests_unavailable'}

    clean_url = sanitize_url(url)
    logging.info("VirusTotal sanitized url: %r", clean_url)

    try:
        headers = {"x-apikey": api_key}
        # Primary submission (POST with form data)
        try:
            r = requests.post("https://www.virustotal.com/api/v3/urls", data={"url": clean_url}, headers=headers, timeout=timeout, proxies=proxies)
        except Exception as e:
            try:
                diag_path = save_raw_response('virustotal_error', clean_url, {'exception': str(e)})
            except Exception:
                diag_path = ''
            logging.warning("VirusTotal POST exception for %r: %s", clean_url, str(e))
            return {'ok': False, 'error': str(e), 'raw_diag': diag_path, 'sanitized': clean_url}

        logging.info("VirusTotal POST status: %s for %r", getattr(r, 'status_code', None), clean_url)

        # If canonicalization error (400), try fallback GET using base64 URL id
        if r.status_code == 400:
            try:
                jerr = r.json()
                err_msg = jerr.get('error', {}).get('message') or str(jerr)
            except Exception:
                err_msg = r.text
            logging.warning("VirusTotal POST returned 400 for %r: %s", clean_url, err_msg)
            try:
                diag_path = save_raw_response('virustotal_error_400', clean_url, {'status': r.status_code, 'text': r.text, 'headers': dict(r.headers)})
            except Exception:
                diag_path = ''

            # Fallback: base64-encode the sanitized URL and GET the URL resource
            try:
                url_id = base64.urlsafe_b64encode(clean_url.encode()).decode().strip('=')
                get_url = f"https://www.virustotal.com/api/v3/urls/{url_id}"
                logging.info("VirusTotal fallback GET url_id=%s for %r", url_id, clean_url)
                try:
                    r_f = requests.get(get_url, headers=headers, timeout=timeout, proxies=proxies)
                except Exception as e2:
                    try:
                        diag_path2 = save_raw_response('virustotal_fallback_error', clean_url, {'exception': str(e2), 'get_url': get_url})
                    except Exception:
                        diag_path2 = ''
                    logging.warning("VirusTotal fallback GET exception for %r: %s", clean_url, str(e2))
                    return {'ok': False, 'error': 'fallback_get_exception', 'message': str(e2), 'raw_diag': diag_path2, 'sanitized': clean_url}

                logging.info("VirusTotal fallback GET status: %s for %r", getattr(r_f, 'status_code', None), clean_url)
                # If the fallback resource is not found, treat it as 'no prior analysis' rather than an error
                if r_f.status_code == 404:
                    try:
                        jerr2 = r_f.json()
                        err_msg2 = jerr2.get('error', {}).get('message') or str(jerr2)
                    except Exception:
                        err_msg2 = r_f.text
                    try:
                        diag_path2 = save_raw_response('virustotal_fallback_not_found', clean_url, {'status': r_f.status_code, 'text': r_f.text, 'headers': dict(r_f.headers), 'get_url': get_url})
                    except Exception:
                        diag_path2 = ''
                    logging.info("VirusTotal fallback GET not found for %r: %s (status %s)", clean_url, err_msg2, r_f.status_code)
                    return {'ok': True, 'summary': 'not_found', 'message': err_msg2, 'raw': r_f.text, 'raw_diag': diag_path2, 'sanitized': clean_url}

                if r_f.status_code != 200:
                    try:
                        jerr2 = r_f.json()
                        err_msg2 = jerr2.get('error', {}).get('message') or str(jerr2)
                    except Exception:
                        err_msg2 = r_f.text
                    try:
                        diag_path2 = save_raw_response('virustotal_fallback_error', clean_url, {'status': r_f.status_code, 'text': r_f.text, 'headers': dict(r_f.headers), 'get_url': get_url})
                    except Exception:
                        diag_path2 = ''
                    logging.warning("VirusTotal fallback GET non-200 for %r: %s (status %s)", clean_url, err_msg2, r_f.status_code)
                    return {'ok': False, 'error': f'fallback_status_{r_f.status_code}', 'message': err_msg2, 'raw_diag': diag_path2, 'sanitized': clean_url}

                jf = r_f.json()
                # stats may be in 'attributes.last_analysis_stats' or 'attributes.stats'
                stats = jf.get('data', {}).get('attributes', {}).get('last_analysis_stats') or jf.get('data', {}).get('attributes', {}).get('stats', {})
                malicious = stats.get('malicious', 0)
                suspicious = stats.get('suspicious', 0)
                harmless = stats.get('harmless', 0)
                summary = 'malicious' if malicious > 0 else ('suspicious' if suspicious > 0 else 'clean')
                return {'ok': True, 'summary': summary, 'malicious': malicious, 'suspicious': suspicious, 'harmless': harmless, 'raw': jf, 'sanitized': clean_url}
            except Exception as e_all:
                logging.exception("VirusTotal fallback flow exception for %r", clean_url)
                return {'ok': False, 'error': str(e_all), 'sanitized': clean_url}

        # Non-400 non-200 responses handled here
        if r.status_code not in (200, 201):
            try:
                jerr = r.json()
                err_msg = jerr.get('error', {}).get('message') or str(jerr)
            except Exception:
                err_msg = r.text
            try:
                diag_path = save_raw_response('virustotal_error', clean_url, {'status': r.status_code, 'text': r.text, 'headers': dict(r.headers)})
            except Exception:
                diag_path = ''
            logging.warning("VirusTotal POST non-200 for %r: %s (status %s)", clean_url, err_msg, r.status_code)
            suggestion = ('400 Bad Request — check the submitted URL is a valid absolute URL, ensure it is not empty or malformed, ' 
                          'and verify your VirusTotal API key and rate limits.')
            return {'ok': False, 'error': f'status_post_{r.status_code}', 'message': err_msg, 'suggestion': suggestion, 'raw': r.text, 'raw_diag': diag_path, 'sanitized': clean_url}

        # Successful POST -> poll analysis
        j = r.json()
        analysis_id = j.get('data', {}).get('id')
        if not analysis_id:
            try:
                diag_path = save_raw_response('virustotal_no_analysis_id', clean_url, {'response': j})
            except Exception:
                diag_path = ''
            logging.warning("VirusTotal POST returned no analysis id for %r", clean_url)
            return {'ok': False, 'error': 'no_analysis_id', 'raw': j, 'raw_diag': diag_path, 'sanitized': clean_url}

        try:
            r2 = requests.get(f"https://www.virustotal.com/api/v3/analyses/{analysis_id}", headers=headers, timeout=timeout, proxies=proxies)
        except Exception as e:
            try:
                diag_path2 = save_raw_response('virustotal_analysis_error', clean_url, {'exception': str(e)})
            except Exception:
                diag_path2 = ''
            logging.warning("VirusTotal analysis GET exception for %r: %s", clean_url, str(e))
            return {'ok': False, 'error': str(e), 'raw_diag': diag_path2, 'sanitized': clean_url}

        logging.info("VirusTotal analysis GET status: %s for %r", getattr(r2, 'status_code', None), clean_url)
        if r2.status_code != 200:
            try:
                jerr2 = r2.json()
                err_msg2 = jerr2.get('error', {}).get('message') or str(jerr2)
            except Exception:
                err_msg2 = r2.text
            try:
                diag_path2 = save_raw_response('virustotal_analysis_error', clean_url, {'status': r2.status_code, 'text': r2.text, 'headers': dict(r2.headers)})
            except Exception:
                diag_path2 = ''
            logging.warning("VirusTotal analysis GET non-200 for %r: %s (status %s)", clean_url, err_msg2, r2.status_code)
            return {'ok': False, 'error': f'status_get_{r2.status_code}', 'message': err_msg2, 'raw': r2.text, 'raw_diag': diag_path2, 'sanitized': clean_url}

        j2 = r2.json()
        stats = j2.get('data', {}).get('attributes', {}).get('stats', {}) or j2.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
        malicious = stats.get('malicious', 0)
        suspicious = stats.get('suspicious', 0)
        harmless = stats.get('harmless', 0)
        summary = 'malicious' if malicious > 0 else ('suspicious' if suspicious > 0 else 'clean')
        return {'ok': True, 'summary': summary, 'malicious': malicious, 'suspicious': suspicious, 'harmless': harmless, 'raw': j2, 'sanitized': clean_url}
    except Exception as e:
        logging.exception("Unexpected VirusTotal error")
        return {'ok': False, 'error': str(e), 'sanitized': clean_url}

        # If canonicalization error (400), try fallback GET using base64 URL id
        if r.status_code == 400:
            try:
                jerr = r.json()
                err_msg = jerr.get('error', {}).get('message') or str(jerr)
            except Exception:
                err_msg = r.text
            logging.warning("VirusTotal POST returned 400 for %r: %s", clean_url, err_msg)
            try:
                diag_path = save_raw_response('virustotal_error_400', clean_url, {'status': r.status_code, 'text': r.text, 'headers': dict(r.headers)})
            except Exception:
                diag_path = ''

            # Fallback: base64-encode the sanitized URL and GET the URL resource
            try:
                url_id = base64.urlsafe_b64encode(clean_url.encode()).decode().strip('=')
                get_url = f"https://www.virustotal.com/api/v3/urls/{url_id}"
                logging.info("VirusTotal fallback GET url_id=%s for %r", url_id, clean_url)
                try:
                    r_f = requests.get(get_url, headers=headers, timeout=timeout, proxies=proxies)
                except Exception as e2:
                    try:
                        diag_path2 = save_raw_response('virustotal_fallback_error', clean_url, {'exception': str(e2), 'get_url': get_url})
                    except Exception:
                        diag_path2 = ''
                    logging.warning("VirusTotal fallback GET exception for %r: %s", clean_url, str(e2))
                    return {'ok': False, 'error': 'fallback_get_exception', 'message': str(e2), 'raw_diag': diag_path2, 'sanitized': clean_url}

                logging.info("VirusTotal fallback GET status: %s for %r", getattr(r_f, 'status_code', None), clean_url)
                # If the fallback resource is not found, treat it as 'no prior analysis' rather than an error
                if r_f.status_code == 404:
                    try:
                        jerr2 = r_f.json()
                        err_msg2 = jerr2.get('error', {}).get('message') or str(jerr2)
                    except Exception:
                        err_msg2 = r_f.text
                    try:
                        diag_path2 = save_raw_response('virustotal_fallback_not_found', clean_url, {'status': r_f.status_code, 'text': r_f.text, 'headers': dict(r_f.headers), 'get_url': get_url})
                    except Exception:
                        diag_path2 = ''
                    logging.info("VirusTotal fallback GET not found for %r: %s (status %s)", clean_url, err_msg2, r_f.status_code)
                    return {'ok': True, 'summary': 'not_found', 'message': err_msg2, 'raw': r_f.text, 'raw_diag': diag_path2, 'sanitized': clean_url}

                if r_f.status_code != 200:
                    try:
                        jerr2 = r_f.json()
                        err_msg2 = jerr2.get('error', {}).get('message') or str(jerr2)
                    except Exception:
                        err_msg2 = r_f.text
                    try:
                        diag_path2 = save_raw_response('virustotal_fallback_error', clean_url, {'status': r_f.status_code, 'text': r_f.text, 'headers': dict(r_f.headers), 'get_url': get_url})
                    except Exception:
                        diag_path2 = ''
                    logging.warning("VirusTotal fallback GET non-200 for %r: %s (status %s)", clean_url, err_msg2, r_f.status_code)
                    return {'ok': False, 'error': f'fallback_status_{r_f.status_code}', 'message': err_msg2, 'raw_diag': diag_path2, 'sanitized': clean_url}

                jf = r_f.json()
                # stats may be in 'attributes.last_analysis_stats' or 'attributes.stats'
                stats = jf.get('data', {}).get('attributes', {}).get('last_analysis_stats') or jf.get('data', {}).get('attributes', {}).get('stats', {})
                malicious = stats.get('malicious', 0)
                suspicious = stats.get('suspicious', 0)
                harmless = stats.get('harmless', 0)
                summary = 'malicious' if malicious > 0 else ('suspicious' if suspicious > 0 else 'clean')
                return {'ok': True, 'summary': summary, 'malicious': malicious, 'suspicious': suspicious, 'harmless': harmless, 'raw': jf, 'sanitized': clean_url}
            except Exception as e_all:
                logging.exception("VirusTotal fallback flow exception for %r", clean_url)
                return {'ok': False, 'error': str(e_all), 'sanitized': clean_url}

        # Non-400 non-200 responses handled here
        if r.status_code not in (200, 201):
            try:
                jerr = r.json()
                err_msg = jerr.get('error', {}).get('message') or str(jerr)
            except Exception:
                err_msg = r.text
            try:
                diag_path = save_raw_response('virustotal_error', clean_url, {'status': r.status_code, 'text': r.text, 'headers': dict(r.headers)})
            except Exception:
                diag_path = ''
            logging.warning("VirusTotal POST non-200 for %r: %s (status %s)", clean_url, err_msg, r.status_code)
            suggestion = ('400 Bad Request — check the submitted URL is a valid absolute URL, ensure it is not empty or malformed, ' 
                          'and verify your VirusTotal API key and rate limits.')
            return {'ok': False, 'error': f'status_post_{r.status_code}', 'message': err_msg, 'suggestion': suggestion, 'raw': r.text, 'raw_diag': diag_path, 'sanitized': clean_url}

        # Successful POST -> poll analysis
        j = r.json()
        analysis_id = j.get('data', {}).get('id')
        if not analysis_id:
            try:
                diag_path = save_raw_response('virustotal_no_analysis_id', clean_url, {'response': j})
            except Exception:
                diag_path = ''
            logging.warning("VirusTotal POST returned no analysis id for %r", clean_url)
            return {'ok': False, 'error': 'no_analysis_id', 'raw': j, 'raw_diag': diag_path, 'sanitized': clean_url}

        try:
            r2 = requests.get(f"https://www.virustotal.com/api/v3/analyses/{analysis_id}", headers=headers, timeout=timeout, proxies=proxies)
        except Exception as e:
            try:
                diag_path2 = save_raw_response('virustotal_analysis_error', clean_url, {'exception': str(e)})
            except Exception:
                diag_path2 = ''
            logging.warning("VirusTotal analysis GET exception for %r: %s", clean_url, str(e))
            return {'ok': False, 'error': str(e), 'raw_diag': diag_path2, 'sanitized': clean_url}

        logging.info("VirusTotal analysis GET status: %s for %r", getattr(r2, 'status_code', None), clean_url)
        if r2.status_code != 200:
            try:
                jerr2 = r2.json()
                err_msg2 = jerr2.get('error', {}).get('message') or str(jerr2)
            except Exception:
                err_msg2 = r2.text
            try:
                diag_path2 = save_raw_response('virustotal_analysis_error', clean_url, {'status': r2.status_code, 'text': r2.text, 'headers': dict(r2.headers)})
            except Exception:
                diag_path2 = ''
            logging.warning("VirusTotal analysis GET non-200 for %r: %s (status %s)", clean_url, err_msg2, r2.status_code)
            return {'ok': False, 'error': f'status_get_{r2.status_code}', 'message': err_msg2, 'raw': r2.text, 'raw_diag': diag_path2, 'sanitized': clean_url}

        j2 = r2.json()
        stats = j2.get('data', {}).get('attributes', {}).get('stats', {}) or j2.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
        malicious = stats.get('malicious', 0)
        suspicious = stats.get('suspicious', 0)
        harmless = stats.get('harmless', 0)
        summary = 'malicious' if malicious > 0 else ('suspicious' if suspicious > 0 else 'clean')
        return {'ok': True, 'summary': summary, 'malicious': malicious, 'suspicious': suspicious, 'harmless': harmless, 'raw': j2, 'sanitized': clean_url}
    except Exception as e:
        logging.exception("Unexpected VirusTotal error")
        return {'ok': False, 'error': str(e), 'sanitized': clean_url}


def query_google_safe_browsing(url, api_key, timeout=10):
    """Query Google Safe Browsing (v4) for a URL. Returns a simple summary and raw response.

    If the API responds with 403 it commonly means the API key is invalid, the Safe Browsing API
    is not enabled on the Google Cloud project, or billing/permissions are required. In that case
    this function includes a helpful 'suggestion' field in the returned dict.
    """
    if not api_key:
        return {'ok': False, 'error': 'no_api_key'}
    if requests is None:
        return {'ok': False, 'error': 'requests_unavailable'}
    try:
        # Use the official Safe Browsing v4 POST endpoint (do not prefix with HTTP verb)
        endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
        payload = {
            "client": {"clientId": "fraud_detection_app", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE","SOCIAL_ENGINEERING","UNWANTED_SOFTWARE","POTENTIALLY_HARMFUL_APPLICATION"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}]
            }
        }
        r = make_request_with_retries('post', endpoint, json=payload, timeout=timeout, max_retries=max_retries, backoff_factor=backoff_factor, proxies=proxies)
        if isinstance(r, dict):
            return {'ok': False, 'error': r.get('error')}
        if r.status_code != 200:
            try:
                jerr = r.json()
                err_msg = jerr.get('error', {}).get('message') or str(jerr)
            except Exception:
                err_msg = r.text
            suggestion = ''
            if r.status_code == 403:
                suggestion = ('403 Forbidden — check your API key, ensure the Google Safe Browsing API is enabled '
                              'for your Google Cloud project and that any required billing/permissions are set up.')
            return {'ok': False, 'error': f'status_{r.status_code}', 'message': err_msg, 'suggestion': suggestion, 'raw': r.text}
        j = r.json()
        matches = j.get('matches')
        if not matches:
            return {'ok': True, 'summary': 'clean', 'matches': [], 'raw': j}
        threats = list({m.get('threatType') for m in matches})
        return {'ok': True, 'summary': 'malicious', 'threats': threats, 'raw': j}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


# def query_urlhaus(url, timeout=10):
#     """Query URLhaus API for URL details. Returns summary and raw response.

#     Uses a friendly User-Agent and sends JSON; when a 401 is returned the function adds a
#     'suggestion' to help diagnose possible IP blocking or auth issues.
#     """
#     if requests is None:
#         return {'ok': False, 'error': 'requests_unavailable'}
#     try:
#         headers = {'User-Agent': 'fraud_detection_app/1.0', 'Accept': 'application/json'}
#         # URLhaus expects form data with key 'url' — avoid sending JSON which some endpoints may reject
#         r = make_request_with_retries('post', 'https://urlhaus-api.abuse.ch/v1/url/', data={'url': url}, headers=headers, timeout=timeout, max_retries=max_retries, backoff_factor=backoff_factor, proxies=proxies)
#         if isinstance(r, dict):
#             return {'ok': False, 'error': r.get('error')}
#         if r.status_code == 401:
#             try:
#                 jerr = r.json()
#                 err_msg = jerr.get('error') or str(jerr)
#             except Exception:
#                 err_msg = r.text
#             # Save full diagnostic for the 401 so you can inspect headers/body later
#             try:
#                 diag_path = save_raw_response('urlhaus_401', url, {'status': r.status_code, 'text': r.text, 'headers': dict(r.headers)})
#             except Exception:
#                 diag_path = ''
#             suggestion = ('401 Unauthorized — URLhaus may be blocking requests from this host or IP. '
#                           'Try again later, check URLhaus API status, or run from a different network.')
#             return {'ok': False, 'error': f'status_{r.status_code}', 'message': err_msg, 'suggestion': suggestion, 'raw': r.text, 'raw_diag': diag_path}
#         if r.status_code != 200:
#             return {'ok': False, 'error': f'status_{r.status_code}', 'raw': r.text}
#         j = r.json()
#         # URLhaus returns 'query_status': 'ok' or 'not_found'
#         if j.get('query_status') == 'ok':
#             return {'ok': True, 'summary': 'malicious', 'raw': j}
#         else:
#             return {'ok': True, 'summary': 'not_found', 'raw': j}
#     except Exception as e:
#         return {'ok': False, 'error': str(e)}


# def query_phishtank(url, api_key=None, timeout=10):
#     """Attempt to query PhishTank for a URL. Requires API key for detailed checks. Returns summary and raw response."""
#     if requests is None:
#         return {'ok': False, 'error': 'requests_unavailable'}
#     # PhishTank's API is rate-limited and requires registration; try a best-effort lookup
#     try:
#         if api_key:
#             r = requests.post('https://checkurl.phishtank.com/checkurl/', data={'url': url, 'format': 'json', 'app_key': api_key}, timeout=timeout)
#             if r.status_code != 200:
#                 return {'ok': False, 'error': f'status_{r.status_code}', 'raw': r.text}
#             j = r.json()
#             results = j.get('results', {})
#             in_database = results.get('in_database')
#             valid = results.get('valid')
#             verified = results.get('verified')
#             summary = 'malicious' if verified else ('suspicious' if in_database else 'clean')
#             return {'ok': True, 'summary': summary, 'raw': j}
#         else:
#             return {'ok': False, 'error': 'no_api_key'}
#     except Exception as e:
#         return {'ok': False, 'error': str(e)}


def append_scan_log(entry, filename='dataset/scan_log.csv'):
    """Append a scan entry (dict) to a CSV file. Creates file with headers if needed."""
    if not os.path.exists('dataset'):
        try:
            os.makedirs('dataset', exist_ok=True)
        except Exception:
            pass
    fieldnames = ['timestamp','type','input','ml_raw_prob','ml_pct','threshold','final_verdict','domain_valid','rule_reasons','phish_match','host_matches_count','virustotal_summary','google_safe_summary','raw_response_files','notes']
    write_header = not os.path.exists(filename)
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: entry.get(k) for k in fieldnames})
    except Exception:
        pass

# --- Professional Cybersecurity Frontend Design ---
# Session state for analytics
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'safe_count' not in st.session_state:
    st.session_state.safe_count = 0
if 'fraud_count' not in st.session_state:
    st.session_state.fraud_count = 0

# Comprehensive modern styling with cybersecurity theme
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0a0e27 0%, #0f0f1e 100%);
    color: #e8eef8;
}

/* === HEADER HERO === */
.header-hero {
    background: linear-gradient(135deg, #1a4d7a 0%, #0d2a4a 50%, #1a1a3a 100%);
    padding: 40px 20px;
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(13, 42, 74, 0.4), inset 0 1px 0 rgba(255,255,255,0.1);
    border: 1px solid rgba(79, 172, 254, 0.2);
}

.header-content {
    display: flex;
    align-items: center;
    gap: 20px;
    max-width: 1000px;
    margin: 0 auto;
}

.header-icon {
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    border-radius: 12px;
    font-size: 32px;
}

.header-text h1 {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-text p {
    font-size: 14px;
    color: #a8b8d8;
    margin-top: 4px;
    font-weight: 500;
}

/* === MAIN CARDS === */
.card {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.8), rgba(10, 15, 30, 0.6));
    border: 1px solid rgba(79, 172, 254, 0.15);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 16px;
}

/* === INPUT SECTION === */
.input-section {
    background: linear-gradient(180deg, rgba(20, 30, 50, 0.9), rgba(15, 20, 35, 0.7));
    border: 2px solid rgba(79, 172, 254, 0.25);
    border-radius: 14px;
    padding: 32px;
    margin: 30px auto;
    max-width: 700px;
    box-shadow: 0 8px 32px rgba(79, 172, 254, 0.1);
}

.input-label {
    font-size: 14px;
    font-weight: 600;
    color: #4facfe;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
    display: block;
}

/* === VERDICT BADGES === */
.verdict-safe {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(74, 222, 128, 0.08));
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #22c55e;
    padding: 16px;
    border-radius: 10px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}

.verdict-fraud {
    background: linear-gradient(135deg, rgba(220, 38, 38, 0.15), rgba(239, 68, 68, 0.08));
    border: 1px solid rgba(220, 38, 38, 0.3);
    color: #ef4444;
    padding: 16px;
    border-radius: 10px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* === PROGRESS BAR === */
.progress-bar-container {
    margin: 16px 0;
}

.progress-label {
    font-size: 13px;
    color: #a8b8d8;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
}

/* === METRICS === */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(180deg, rgba(20, 30, 50, 0.6), rgba(15, 20, 35, 0.4));
    border: 1px solid rgba(79, 172, 254, 0.15);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.metric-value {
    font-size: 24px;
    font-weight: 800;
    color: #4facfe;
    margin-bottom: 4px;
}

.metric-label {
    font-size: 12px;
    color: #6b7a9a;
    font-weight: 500;
    text-transform: uppercase;
}

/* === BADGES === */
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
}

.badge-safe {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.4);
}

.badge-fraud {
    background: rgba(220, 38, 38, 0.2);
    color: #ef4444;
    border: 1px solid rgba(220, 38, 38, 0.4);
}

.badge-trusted {
    background: rgba(34, 197, 94, 0.2);
    color: #10b981;
    border: 1px solid rgba(34, 197, 94, 0.4);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* === FOOTER === */
.footer {
    text-align: center;
    color: #6b7a9a;
    font-size: 12px;
    margin-top: 60px;
    padding: 20px;
    border-top: 1px solid rgba(79, 172, 254, 0.1);
}

/* === UTILITIES === */
.text-muted { color: #6b7a9a; }
.text-info { color: #4facfe; }
.divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(79, 172, 254, 0.1) 0%, rgba(79, 172, 254, 0.3) 50%, rgba(79, 172, 254, 0.1) 100%);
    margin: 24px 0;
}

</style>
""", unsafe_allow_html=True)

# Header Hero Banner
st.markdown("""
<div class='header-hero'>
    <div class='header-content'>
        <div class='header-icon'>🛡️</div>
        <div class='header-text'>
            <h1>AI Cyber Fraud Detector</h1>
            <p>Real-time URL, Message & Scam Analysis • ML + Rules + External Checks</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ML threshold control (default safer value)
import json
recommended_default = 0.5
try:
    with open('model/url_model_calibration_info.json','r',encoding='utf-8') as _fh:
        _info = json.load(_fh)
        if _info and isinstance(_info.get('recommended_threshold'), (int,float)):
            recommended_default = float(_info.get('recommended_threshold'))
except Exception:
    _info = None

try:
    ml_threshold = float(st.sidebar.slider('ML fraud probability threshold', min_value=0.0, max_value=1.0, value=recommended_default, step=0.01))
except Exception:
    ml_threshold = recommended_default

# Indicate whether a calibrated model is used and show calibration metrics when available
if 'calibrated_model_used' in globals() and calibrated_model_used:
    st.sidebar.success('Calibrated URL model loaded (safer probabilities)')
else:
    st.sidebar.info('Standard URL model loaded')

if _info:
    try:
        st.sidebar.markdown('**Calibration metrics:**')
        if _info.get('brier_score') is not None:
            st.sidebar.write(f"Brier score: {_info.get('brier_score'):.4f}")
        if _info.get('roc_auc') is not None:
            st.sidebar.write(f"ROC AUC: {_info.get('roc_auc'):.3f}")
        # Show recommended threshold and indicate if it was clamped for safety
        rec = _info.get('recommended_threshold')
        st.sidebar.write(f"Recommended threshold: {rec:.2f}")
        if _info.get('threshold_clamped'):
            st.sidebar.warning(f"Note: recommended threshold was clamped to {rec:.2f} for safety (original: {_info.get('original_recommended_threshold'):.2f})")
    except Exception:
        pass

# Detect and optionally load a lightweight (faster) URL model saved during training
light_model_available = False
url_model_light = None
try:
    if os.path.exists('model/url_model_light.pkl'):
        url_model_light = pickle.load(open('model/url_model_light.pkl','rb'))
        light_model_available = True
except Exception:
    light_model_available = False

# Runtime selection: allow user to choose lightweight model for faster inference
runtime_url_model = url_model
use_light = False
if light_model_available:
    use_light = st.sidebar.checkbox('Use lightweight URL model for faster inference', value=False)
    if use_light and url_model_light is not None:
        runtime_url_model = url_model_light

if _info and _info.get('model_name'):
    try:
        model_name = _info.get('model_name')
        model_label = f"{model_name} (light)" if use_light else model_name
        st.sidebar.write(f"Model: {model_label}")
    except Exception:
        pass

# --- API keys configuration ---
VT_API_KEY_DEFAULT = "" # Enter your VirusTotal API here or into the site (Optional)
GSB_API_KEY_DEFAULT = "" # Enter your google safe browsing API here or into the site (Optional)
TRUSTED_DOMAINS = {
    'google.com','youtube.com','microsoft.com','github.com','amazon.com','facebook.com'
}
TRUSTED_DOMAIN_OVERRIDE_CAP = 0.10 
# Calibration and scoring defaults
CONFIDENCE_DAMPEN_FACTOR = 0.5  
WEIGHTS = {'ml': 0.6, 'rule': 0.3, 'reputation': 0.1}


def host_in_trusted(host: str, local_whitelist: set = None) -> bool:
    """Return True if host matches a built-in trusted domain or an entry in a local whitelist.

    Matching is permissive: exact match or endswith('.domain') is considered trusted (so
    'www.google.com' matches 'google.com').
    """
    if not host:
        return False
    host = host.lower()
    # check built-in trusted domains
    for d in TRUSTED_DOMAINS:
        if host == d or host.endswith('.' + d):
            return True
    # check local whitelist set if provided
    if local_whitelist:
        for d in local_whitelist:
            if not d:
                continue
            d = d.lower()
            if host == d or host.endswith('.' + d):
                return True
    return False

# External site API keys (optional) — enable checks on the main page using the checkboxes
vt_api_key = st.sidebar.text_input('VirusTotal API key (optional) — enter to enable main-page VT check', value=os.getenv('VT_API_KEY') or VT_API_KEY_DEFAULT, type='password')
gsb_api_key = st.sidebar.text_input('Google Safe Browsing API key (optional) — enter to enable main-page GSB check', value=os.getenv('GSB_API_KEY') or GSB_API_KEY_DEFAULT, type='password')
# phishtank_api_key = st.sidebar.text_input('PhishTank API key (optional) — enter to enable main-page PhishTank check', value=os.getenv('PHISHTANK_API_KEY') or PHISHTANK_API_KEY_DEFAULT, type='password')

# Proxy & retry controls (optional)
use_proxy = st.sidebar.checkbox('Use HTTP proxy for external checks', value=False)
proxy_url = st.sidebar.text_input('Proxy URL (e.g. http://127.0.0.1:8080)', value=os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY') or "")
proxy_username = st.sidebar.text_input('Proxy username (optional)', value='')
proxy_password = st.sidebar.text_input('Proxy password (optional)', value='', type='password')

max_retries = int(st.sidebar.number_input('External check max retries', min_value=0, max_value=10, value=3, step=1))
backoff_factor = float(st.sidebar.number_input('Backoff factor (seconds)', min_value=0.0, max_value=10.0, value=0.5, step=0.1, format="%.2f"))

# Custom User-Agent for external checks (can help avoid 401 blocks)
custom_user_agent = st.sidebar.text_input('Custom User-Agent (optional)', value='fraud_detection_app/1.0')

if use_proxy and proxy_url:
    st.sidebar.write(f"Proxy enabled: {proxy_url}")
elif use_proxy and not proxy_url:
    st.sidebar.warning('Proxy enabled but no Proxy URL set; please enter a proxy or disable the option')

# Build proxies dict for requests if requested; include credentials if provided
proxy_url_with_auth = proxy_url
if use_proxy and proxy_url and proxy_username:
    # Insert credentials into proxy URL: http://user:pass@host:port
    from urllib.parse import urlparse, urlunparse
    p = urlparse(proxy_url)
    netloc = f"{proxy_username}:{proxy_password}@{p.hostname}"
    if p.port:
        netloc += f":{p.port}"
    proxy_url_with_auth = urlunparse((p.scheme, netloc, p.path or '', p.params or '', p.query or '', p.fragment or ''))

proxies = {'http': proxy_url_with_auth, 'https': proxy_url_with_auth} if use_proxy and proxy_url_with_auth else None

# Logging option
enable_logging = st.sidebar.checkbox('Enable logging of scans to dataset/scan_log.csv', value=True)

# Log viewer option
show_logs = st.sidebar.checkbox('Show recent scan logs (with filters)', value=False)

# Inform where to enter API keys
if not (vt_api_key or os.getenv('VT_API_KEY') or VT_API_KEY_DEFAULT):
    st.sidebar.info('If you want to use VirusTotal checks, enter a VT API key here or paste it into the VT_API_KEY_DEFAULT constant at the top of the file.')
if not (gsb_api_key or os.getenv('GSB_API_KEY') or GSB_API_KEY_DEFAULT):
    st.sidebar.info('If you want to use Google Safe Browsing checks, enter a GSB API key here or paste it into the GSB_API_KEY_DEFAULT constant at the top of the file.')
# if not (phishtank_api_key or os.getenv('PHISHTANK_API_KEY') or PHISHTANK_API_KEY_DEFAULT):
#     st.sidebar.info('If you want to use PhishTank checks, enter a PhishTank API key here or paste it into the PHISHTANK_API_KEY_DEFAULT constant at the top of the file.')

# If we loaded verified phishing data, show a small summary in the sidebar
# if phish_lookup:
    try:
        import pandas as _pd
        _df = _pd.DataFrame(phish_lookup.values())
        st.sidebar.subheader('Verified Phishing Dataset')
        st.sidebar.write(f"Records: {_df.shape[0]}")
        top_targets = _df['target'].fillna('Other').value_counts().head(10)
        st.sidebar.write('Top targets:')
        for t, c in top_targets.items():
            st.sidebar.write(f"- {t}: {c}")
        if 'submission_time' in _df.columns:
            st.sidebar.write(f"Date range: {_df['submission_time'].min()} → {_df['submission_time'].max()}")
    except Exception:
        pass
# st.markdown("Detect fraudulent messages and URLs using advanced machine learning models.")
st.markdown("Advanced fraud detection using Machine Learning")

# Modern navigation with tabs
tab1, tab2, tab3 = st.tabs(["🔗 URL Check", "💬 Message Check", "📊 Analytics"])

with tab1:
    # Input section with professional styling
    st.markdown("""
    <div class='input-section'>
        <label class='input-label'>Enter URL to analyze</label>
    </div>
    """, unsafe_allow_html=True)
    
    url_raw = st.text_input("", placeholder="https://example.com/path", key='url_input', label_visibility="collapsed")
    url = url_raw.strip()  # Trim whitespace
    
    # Check options row
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        vt_check = st.checkbox('🛡️ VirusTotal', value=False)
    with col2:
        gsb_check = st.checkbox('🔎 Safe Browsing', value=False)
    with col3:
        st.markdown("")  # spacer
    
    # Scan button
    scan_btn = st.button("▶️ Scan Now", use_container_width=True, key="scan_url_btn")
    
    # Process URL submission
    if scan_btn:
        if url:  # Check if not empty after trimming
            # Auto-normalize URL by adding https:// if no scheme present
            if not url.startswith(('http://', 'https://')):
                normalized = 'https://' + url
            else:
                normalized = url
            
            with st.spinner("🔍 Analyzing URL..."):
                from dataset.utils.url_normalize import normalize_url
                from dataset.utils.url_rules import rule_check
                
                # 1. Normalize URL
                info = normalize_url(normalized)
                normalized_url = info.get('normalized_url', '')
                
                # 2. Rule-based deterministic checks
                rule_flag, rule_reasons, domain_valid, unusual_findings = rule_check(info)
                
                # Host for whitelist and reputation checks
                host = info.get('netloc','').lower()
                
                # 3. ML probability
                model_input_url = prepare_url_for_model(normalized_url)
                features = [extract_url_features(model_input_url)]
                
                try:
                    probs = runtime_url_model.predict_proba(features)[0]
                except Exception as e:
                    logging.warning("predict_proba failed: %s", str(e))
                    pred = runtime_url_model.predict(features)[0]
                    probs = [0.0, 1.0] if pred == 1 else [1.0, 0.0]
                
                try:
                    fraud_index = list(runtime_url_model.classes_).index(1)
                    ml_raw_prob = float(probs[fraud_index])
                except Exception:
                    ml_raw_prob = float(max(probs))
                
                # Compute rule-based penalty
                penalty, rb_high_risk, rb_details = compute_rule_risk(info, rule_reasons)
                rule_score = min(1.0, penalty / 0.6) if penalty > 0 else 0.0
                
                # Apply confidence dampening
                ml_calibrated = ml_raw_prob
                calibration_notes = []
                if domain_valid and (not rule_flag) and penalty == 0 and (not info.get('is_ip')):
                    ml_calibrated = ml_raw_prob * CONFIDENCE_DAMPEN_FACTOR
                    calibration_notes.append(f"dampened_by_confidence(x{CONFIDENCE_DAMPEN_FACTOR})")
                
                reputation_score = 0.0
                
                # Initial weighted score
                combined_score = WEIGHTS['ml'] * ml_calibrated + WEIGHTS['rule'] * rule_score + WEIGHTS['reputation'] * reputation_score
                
                if rb_high_risk and combined_score < 0.30:
                    combined_score = 0.30
                
                fraud_pct = combined_score * 100.0
                safe_pct = 100.0 - fraud_pct
                final_fraud_base = (combined_score >= ml_threshold) or rule_flag or (not domain_valid)
                
                # External checks
                vt_result = None
                gsb_result = None
                raw_files = []
                
                if vt_check:
                    api_key = vt_api_key or os.getenv('VT_API_KEY') or VT_API_KEY_DEFAULT
                    if api_key:
                        vt_result = query_virustotal(normalized_url, api_key)
                        if vt_result.get('ok'):
                            summary = vt_result.get('summary')
                            malicious = vt_result.get('malicious', 0)
                            suspicious = vt_result.get('suspicious', 0)
                            if summary == 'malicious':
                                st.markdown(f"<span class='badge badge-red'>VirusTotal: MALICIOUS ({malicious} engines)</span>", unsafe_allow_html=True)
                            elif summary == 'suspicious':
                                st.markdown(f"<span class='badge badge-amber'>VirusTotal: SUSPICIOUS ({suspicious} engines)</span>", unsafe_allow_html=True)
                            elif summary == 'not_found':
                                st.markdown(f"<span class='badge badge-amber'>VirusTotal: NOT FOUND (no prior analysis)</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span class='badge badge-green'>VirusTotal: CLEAN</span>", unsafe_allow_html=True)
                            if vt_result.get('raw') is not None:
                                path = save_raw_response('virustotal', normalized_url, vt_result.get('raw'))
                                if path:
                                    raw_files.append(path)
                        else:
                            st.error(f"VirusTotal check failed: {vt_result.get('error')}")
                
                if gsb_check:
                    api_key = gsb_api_key or os.getenv('GSB_API_KEY') or GSB_API_KEY_DEFAULT
                    if api_key:
                        gsb_result = query_google_safe_browsing(normalized_url, api_key)
                        if gsb_result.get('ok'):
                            st.write(f"Google Safe Browsing: {gsb_result.get('summary')}")
                            path = save_raw_response('google_safebrowsing', normalized_url, gsb_result.get('raw'))
                            if path:
                                raw_files.append(path)
                        else:
                            st.error(f"Google Safe Browsing check failed: {gsb_result.get('error')}")
                
                # Load whitelist and compute final adjusted score
                local_whitelist = set()
                try:
                    with open('dataset/whitelist.txt','r',encoding='utf-8') as fh:
                        for line in fh:
                            local_whitelist.add(line.strip().lower())
                except Exception:
                    local_whitelist = set()
                
                trusted_override = False
                if host_in_trusted(host, local_whitelist):
                    trusted_override = True
                    reputation_score = 0.0
                else:
                    if vt_result and vt_result.get('ok'):
                        s = vt_result.get('summary')
                        if s == 'malicious':
                            reputation_score = 1.0
                        elif s == 'suspicious':
                            reputation_score = 0.6
                    if gsb_result and gsb_result.get('ok'):
                        if gsb_result.get('summary') == 'malicious':
                            reputation_score = max(reputation_score, 1.0)
                
                # Final adjusted score
                final_adjusted = WEIGHTS['ml'] * ml_calibrated + WEIGHTS['rule'] * rule_score + WEIGHTS['reputation'] * reputation_score
                
                if trusted_override:
                    final_adjusted = min(final_adjusted, TRUSTED_DOMAIN_OVERRIDE_CAP)
                
                if rb_high_risk and final_adjusted < 0.30:
                    final_adjusted = 0.30
                
                fraud_pct = final_adjusted * 100.0
                safe_pct = 100.0 - fraud_pct
                final_fraud = (final_adjusted >= ml_threshold) or rule_flag or (not domain_valid)
                
                if trusted_override:
                    st.info("Trusted domain – ML score overridden")
                
                # Save results to log
                if enable_logging:
                    log_entry = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'type': 'URL',
                        'input': normalized_url,
                        'ml_raw_prob': float(final_adjusted),
                        'ml_pct': f"{final_adjusted*100:.2f}%",
                        'threshold': float(ml_threshold),
                        'final_verdict': 'FRAUD' if final_fraud else 'SAFE',
                        'domain_valid': domain_valid,
                        'rule_reasons': ';'.join(rule_reasons) if rule_reasons else '',
                        'phish_match': '',
                        'host_matches_count': '',
                        'virustotal_summary': vt_result.get('summary') if vt_result else '',
                        'google_safe_summary': gsb_result.get('summary') if gsb_result else '',
                        'raw_response_files': ';'.join(raw_files) if raw_files else '',
                        'notes': f"ml_raw={ml_raw_prob:.4f};ml_calibrated={ml_calibrated:.4f};penalty={penalty:.3f};trusted_override={trusted_override}"
                    }
                    append_scan_log(log_entry)
                
                # Display results
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='card'><div class='card-title'>📊 Analysis Results</div>", unsafe_allow_html=True)
                
                if final_fraud:
                    st.markdown(f"<div class='verdict-fraud'>❌ FRAUD DETECTED</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='verdict-safe'>✅ SAFE</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='progress-bar-container'>", unsafe_allow_html=True)
                st.markdown("<div class='progress-label'><span>Fraud Probability</span><span>" + f"{fraud_pct:.1f}%" + "</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.progress(min(max(final_adjusted, 0.0), 1.0))
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Raw ML Score", f"{ml_raw_prob*100:.1f}%")
                with m2:
                    st.metric("Adjusted Score", f"{final_adjusted*100:.1f}%")
                with m3:
                    st.metric("Decision Threshold", f"{ml_threshold*100:.1f}%")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='card'><div class='card-title'>🔐 Domain Validation</div>", unsafe_allow_html=True)
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    if domain_valid:
                        st.markdown("<span class='badge badge-safe'>✅ Valid Domain</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge badge-fraud'>❌ Invalid Domain</span>", unsafe_allow_html=True)
                with dcol2:
                    if trusted_override:
                        st.markdown("<span class='badge badge-trusted'>⭐ Trusted Domain</span>", unsafe_allow_html=True)
                
                if rule_reasons:
                    with st.expander("📋 Rule-Based Detections"):
                        for reason in rule_reasons:
                            st.write(f"• {reason}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                with st.expander("🤖 ML & Scoring Details"):
                    st.markdown(f"""
                    <div class='card'>
                    <strong>Probability Breakdown:</strong><br>
                    • Raw ML Score: {ml_raw_prob*100:.2f}%<br>
                    • Calibrated ML: {ml_calibrated*100:.2f}%<br>
                    • Rule Penalty: {penalty*100:.1f}%<br>
                    • Reputation Score: {reputation_score*100:.1f}%<br>
                    • <strong>Final Adjusted: {final_adjusted*100:.2f}%</strong><br>
                    <br>
                    <strong>Scoring Weights:</strong><br>
                    • ML Model: 60%<br>
                    • Rule-Based: 30%<br>
                    • Reputation/Whitelist: 10%
                    </div>
                    """, unsafe_allow_html=True)
                
                st.session_state.total_scans += 1
                if final_fraud:
                    st.session_state.fraud_count += 1
                else:
                    st.session_state.safe_count += 1
                
                st.markdown(f"**Normalized URL:** `{normalized_url}`", unsafe_allow_html=True)
                st.markdown("---")
                if final_fraud:
                    st.error("FINAL VERDICT: ⚠️ FRAUD")
                else:
                    st.success("FINAL VERDICT: ✅ SAFE")
        else:
            st.warning("⚠️ Please enter a URL to analyze")

with tab2:
    st.markdown("""
    <div class='input-section'>
        <label class='input-label'>Enter message to analyze</label>
    </div>
    """, unsafe_allow_html=True)
    
    msg = st.text_area("", placeholder="Paste suspicious message here...", key='msg_input', label_visibility="collapsed", height=150)
    scan_msg_btn = st.button("▶️ Scan Message", use_container_width=True, key="scan_msg_btn")
    
    # Process message submission
    if scan_msg_btn and msg.strip():
        with st.spinner("🔍 Analyzing message..."):
            from dataset.utils.text_clean import clean_text
            cleaned_msg = clean_text(msg)
            vec = vectorizer.transform([cleaned_msg])
            
            # Get prediction and confidence (show exact probability)
            prediction = text_model.predict(vec)[0]
            proba_raw = max(text_model.predict_proba(vec)[0])
            confidence_pct = proba_raw * 100
            
            # Display result with professional styling
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='card'><div class='card-title'>📋 Analysis Result</div>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown(f"<div class='verdict-fraud'>❌ FRAUD DETECTED</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='verdict-safe'>✅ SAFE</div>", unsafe_allow_html=True)
            
            # Show confidence metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence Score", f"{confidence_pct:.2f}%")
            with col2:
                st.metric("Fraud Probability", f"{proba_raw:.4f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Update session metrics
            st.session_state.total_scans += 1
            if prediction == 0:
                st.session_state.safe_count += 1
            else:
                st.session_state.fraud_count += 1
            
            # Log message prediction when enabled
            if enable_logging:
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': 'Message',
                    'input': cleaned_msg[:100],
                    'ml_raw_prob': float(proba_raw),
                    'ml_pct': f"{proba_raw*100:.2f}%",
                    'threshold': '',
                    'final_verdict': 'FRAUD' if prediction == 1 else 'SAFE',
                    'domain_valid': '',
                    'rule_reasons': '',
                    'phish_match': '',
                    'host_matches_count': '',
                    'virustotal_summary': '',
                    'google_safe_summary': '',
                    'notes': ''
                }
                append_scan_log(log_entry)

with tab3:
    st.markdown("<div class='card'><div class='card-title'>📈 Scan Analytics</div>", unsafe_allow_html=True)
    
    # Display metrics
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Total Scans", st.session_state.total_scans, delta=None)
    with m_col2:
        st.metric("Safe URLs", st.session_state.safe_count, delta=None)
    with m_col3:
        st.metric("Fraud Detected", st.session_state.fraud_count, delta=None)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show logs viewer option
    show_logs_opt = st.checkbox('Show recent scan logs', value=False)
    if show_logs_opt:
        try:
            import pandas as pd
            logs_path = 'dataset/scan_log.csv'
            if os.path.exists(logs_path):
                df = pd.read_csv(logs_path, parse_dates=['timestamp'])
                st.dataframe(df.sort_values('timestamp', ascending=False).head(10), use_container_width=True)
            else:
                st.info('No scan logs found yet.')
        except Exception as e:
            st.error(f'Error loading logs: {e}')

# URL processing logic is now inside tab1 - see above
        else:
            st.warning("⚠️ Please enter a URL to analyze")

# ---- Logs viewer section (shown when chosen in sidebar) ---
if show_logs:
    st.header('Recent scan logs')
    try:
        import pandas as pd
        logs_path = 'dataset/scan_log.csv'
        if os.path.exists(logs_path):
            df = pd.read_csv(logs_path, parse_dates=['timestamp'])
            # Sidebar filters for logs viewer
            st.subheader('Filters')
            f_type = st.selectbox('Type', options=['All'] + sorted(df['type'].dropna().unique().tolist()), index=0)
            f_verdict = st.selectbox('Final verdict', options=['All'] + sorted(df['final_verdict'].dropna().unique().tolist()), index=0)
            min_prob = st.slider('Minimum ML probability (%)', min_value=0.0, max_value=100.0, value=0.0)
            search = st.text_input('Search input (URL or message)', '')

            filt = df.copy()
            if f_type != 'All':
                filt = filt[filt['type'] == f_type]
            if f_verdict != 'All':
                filt = filt[filt['final_verdict'] == f_verdict]
            if min_prob > 0:
                try:
                    filt = filt[filt['ml_raw_prob'] >= (min_prob/100.0)]
                except Exception:
                    pass
            if search.strip():
                filt = filt[filt['input'].str.contains(search.strip(), na=False, case=False)]

            st.write(f"Showing {len(filt)} / {len(df)} logs")
            st.dataframe(filt.sort_values('timestamp', ascending=False).reset_index(drop=True))

            # Offer export
            if st.button('Export filtered logs to CSV'):
                outpath = 'dataset/scan_log_export.csv'
                filt.to_csv(outpath, index=False)
                st.success(f'Filtered logs saved to {outpath}')
        else:
            st.info('No scan logs found yet. Enable logging in the sidebar to start saving scans.')
    except Exception as e:
        st.error(f'Failed to load logs: {e}')
# ----- Footer ---
st.markdown("""
<div class='footer'>
    <p>Built with <span style='color:#ef4444'>❤️</span> for Cyber Security Hackathon</p>
    <p style='font-size:11px; color:#4b5563; margin-top:8px;'>
        <strong>Disclaimer:</strong> ML predictions are probabilistic and not 100% accurate. Always verify suspicious content through multiple sources.
        External checks depend on third-party API availability.
    </p>
    <p style='font-size:10px; color:#3a414a; margin-top:8px;'>v1.0 • AI Cyber Fraud Detector • Powered by scikit-learn & Streamlit</p>
</div>

""", unsafe_allow_html=True)
