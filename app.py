#!/usr/bin/env python3
"""Stock Portfolio Tracker - Flask Backend"""
import json
import traceback
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import yfinance as yf
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# yfinance wrapper with fallback for cloud servers
import pandas as pd
import requests as http_requests

# Try to create yfinance with curl_cffi; fall back to raw Yahoo API if unavailable
# Environment override: set _YF_WORKS=False to force fallback (for testing)
_YF_WORKS = os.environ.get('_YF_WORKS', '').lower() != 'false'
if _YF_WORKS:
    try:
        _test = yf.Ticker('AAPL')
        _hist = _test.history(period='5d')
        if _hist is None or _hist.empty:
            raise RuntimeError("yfinance returned empty data")
        print("yfinance works OK")
    except Exception as e:
        _YF_WORKS = False
        print(f"yfinance unavailable ({e}) — using raw Yahoo Finance API fallback")
else:
    print("_YF_WORKS=False set — using raw Yahoo Finance API fallback")

_YF_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Yahoo session with crumb for v10 API
_yahoo_session = None
_yahoo_crumb = None
_yahoo_crumb_ts = 0  # timestamp of last crumb fetch

def _init_yahoo_session(force_refresh=False):
    """Initialize Yahoo session and get crumb. Refreshes if older than 30 min."""
    global _yahoo_session, _yahoo_crumb, _yahoo_crumb_ts
    import time as _t
    if not force_refresh and _yahoo_session and _yahoo_crumb and (_t.time() - _yahoo_crumb_ts) < 1800:
        return
    _yahoo_session = http_requests.Session()
    # Rotate user agents to reduce blocking
    _agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    ]
    import random
    _yahoo_session.headers['User-Agent'] = random.choice(_agents)
    _yahoo_session.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    _yahoo_session.headers['Accept-Language'] = 'en-US,en;q=0.9'
    try:
        # Get cookies from Yahoo
        try:
            _yahoo_session.get('https://fc.yahoo.com', timeout=10, allow_redirects=True)
        except Exception:
            pass  # fc.yahoo.com returns 404 but sets cookies — that's fine

        # Try multiple crumb endpoints
        crumb_urls = [
            'https://query2.finance.yahoo.com/v1/test/getcrumb',
            'https://query1.finance.yahoo.com/v1/test/getcrumb',
        ]
        for crumb_url in crumb_urls:
            for _attempt in range(3):
                try:
                    r = _yahoo_session.get(crumb_url, timeout=10)
                    if r.status_code == 429:
                        _t.sleep(2 ** _attempt + 1)
                        continue
                    if r.status_code == 200 and r.text and len(r.text) < 100:
                        _yahoo_crumb = r.text
                        _yahoo_crumb_ts = _t.time()
                        print(f"Yahoo crumb obtained from {crumb_url}: {_yahoo_crumb[:8]}...")
                        return
                except Exception:
                    _t.sleep(1)
                    continue
        print("Failed to get Yahoo crumb from any endpoint")
        _yahoo_crumb = None
    except Exception as e:
        print(f"Failed to get Yahoo crumb: {e}")
        _yahoo_crumb = None

def _yahoo_get_with_retry(url, headers=None, session=None, params=None, timeout=15, max_retries=3):
    """GET with retry on 429 rate-limit errors."""
    import time as _t
    requester = session or http_requests
    last_r = None
    for attempt in range(max_retries):
        try:
            r = requester.get(url, headers=headers, params=params, timeout=timeout)
            last_r = r
            if r.status_code == 429:
                wait = 2 ** attempt + 2
                print(f"Rate limited (429), waiting {wait}s before retry {attempt+1}/{max_retries}")
                _t.sleep(wait)
                continue
            return r
        except Exception as e:
            print(f"_yahoo_get_with_retry attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                _t.sleep(2 ** attempt + 1)
                continue
            return last_r  # return last response (or None)
    return r  # return last response even if 429

def _raw_yahoo_info(symbol):
    """Fallback: fetch stock info directly from Yahoo Finance v8 chart API."""
    import urllib.parse
    encoded = urllib.parse.quote(symbol, safe='')
    # Try both query1 and query2
    for host in ['query1', 'query2']:
        url = f'https://{host}.finance.yahoo.com/v8/finance/chart/{encoded}?interval=1d&range=5d'
        try:
            r = _yahoo_get_with_retry(url, headers=_YF_HEADERS, timeout=15)
            if r is None or r.status_code != 200:
                continue
            data = r.json()
            result = data.get('chart', {}).get('result', [])
            if result:
                meta = result[0].get('meta', {})
                if meta.get('regularMarketPrice'):
                    return meta
        except Exception as e:
            print(f"_raw_yahoo_info({symbol}) via {host}: {e}")
            continue
    print(f"_raw_yahoo_info({symbol}): all endpoints failed")
    return {}

def _raw_yahoo_quote(symbol):
    """Fallback: fetch quote summary from Yahoo Finance v10 with crumb."""
    import urllib.parse
    encoded = urllib.parse.quote(symbol, safe='')
    for attempt in range(2):
        _init_yahoo_session(force_refresh=(attempt > 0))
        if not _yahoo_crumb:
            return {}
        url = f'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{encoded}'
        params = {
            'modules': 'price,summaryDetail,financialData,defaultKeyStatistics',
            'crumb': _yahoo_crumb,
        }
        try:
            r = _yahoo_get_with_retry(url, session=_yahoo_session, params=params, timeout=15)
            if r.status_code in (401, 403):
                # Crumb expired, refresh and retry
                continue
            if r.status_code != 200:
                print(f"_raw_yahoo_quote({symbol}): HTTP {r.status_code}")
                return {}
            data = r.json()
            result = data.get('quoteSummary', {}).get('result', [{}])[0]
            return result
        except Exception as e:
            print(f"_raw_yahoo_quote({symbol}) attempt {attempt}: {e}")
            if attempt == 0:
                continue
    return {}

def _raw_yahoo_history(symbol, period='5y', interval='1d'):
    """Fallback: fetch price history from Yahoo Finance chart API."""
    import urllib.parse
    range_map = {'1mo': '1mo', '3mo': '3mo', '6mo': '6mo', '1y': '1y', '2y': '2y', '5y': '5y', '10y': '10y', '5d': '5d'}
    r_range = range_map.get(period, '5y')
    encoded = urllib.parse.quote(symbol, safe='')
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?interval={interval}&range={r_range}'
    try:
        r = _yahoo_get_with_retry(url, headers=_YF_HEADERS, timeout=20)
        if r is None or r.status_code != 200:
            print(f"_raw_yahoo_history({symbol}): HTTP {r.status_code if r else 'None'}")
            return pd.DataFrame()
        data = r.json()
        result = data.get('chart', {}).get('result', [])
        if not result:
            return pd.DataFrame()
        timestamps = result[0].get('timestamp', [])
        indicators = result[0].get('indicators', {}).get('quote', [{}])[0]
        closes = indicators.get('close', [])
        volumes = indicators.get('volume', [])
        highs = indicators.get('high', [])
        lows = indicators.get('low', [])
        opens = indicators.get('open', [])

        if not timestamps:
            return pd.DataFrame()

        from datetime import datetime as _dt
        dates = [_dt.fromtimestamp(ts) for ts in timestamps]
        df = pd.DataFrame({
            'Open': opens, 'High': highs, 'Low': lows,
            'Close': closes, 'Volume': volumes,
        }, index=pd.DatetimeIndex(dates))
        df = df.dropna(subset=['Close'])
        return df
    except Exception as e:
        print(f"_raw_yahoo_history({symbol}): {e}")
        return pd.DataFrame()

def yf_ticker(symbol):
    """Create a yfinance Ticker — or return a fallback wrapper."""
    if _YF_WORKS:
        return yf.Ticker(symbol)
    # Return a simple wrapper that uses raw API
    class _FallbackTicker:
        def __init__(self, sym):
            self.ticker = sym
            self._info = None
        @property
        def info(self):
            if self._info is None:
                try:
                    # v8 chart API always works (no crumb needed)
                    meta = _raw_yahoo_info(self.ticker)
                    if not meta or not meta.get('regularMarketPrice'):
                        print(f"_FallbackTicker({self.ticker}): v8 chart returned no data")
                        self._info = {'regularMarketPrice': None}
                        return self._info

                    # Start with v8 data (always available)
                    self._info = {
                        'shortName': meta.get('shortName', self.ticker),
                        'longName': meta.get('longName', meta.get('shortName', self.ticker)),
                        'currentPrice': meta.get('regularMarketPrice', 0),
                        'regularMarketPrice': meta.get('regularMarketPrice', 0),
                        'previousClose': meta.get('previousClose', meta.get('chartPreviousClose', 0)),
                        'currency': meta.get('currency', 'USD'),
                        'financialCurrency': meta.get('currency', 'USD'),
                        'exchangeName': meta.get('exchangeName', ''),
                        'regularMarketVolume': meta.get('regularMarketVolume'),
                        'fiftyTwoWeekHigh': meta.get('fiftyTwoWeekHigh'),
                        'fiftyTwoWeekLow': meta.get('fiftyTwoWeekLow'),
                    }

                    # Enrich with search API (no crumb needed — sector/industry)
                    try:
                        sr = _yahoo_get_with_retry(
                            f'https://query1.finance.yahoo.com/v1/finance/search?q={self.ticker}&quotesCount=1&newsCount=0',
                            headers=_YF_HEADERS, timeout=10)
                        if sr and sr.status_code == 200:
                            sq = sr.json().get('quotes', [{}])[0]
                            self._info['sector'] = sq.get('sector', '')
                            self._info['industry'] = sq.get('industry', '')
                            if sq.get('longname'):
                                self._info['longName'] = sq['longname']
                    except Exception:
                        pass

                    # Try to enrich with v10 quoteSummary (needs crumb, may fail)
                    try:
                        quote = _raw_yahoo_quote(self.ticker)
                        if quote:
                            price_data = quote.get('price', {})
                            summary = quote.get('summaryDetail', {})
                            financial = quote.get('financialData', {})
                            stats = quote.get('defaultKeyStatistics', {})
                            enriched = {
                                'marketCap': self._raw_val(price_data.get('marketCap')),
                                'trailingPE': self._raw_val(summary.get('trailingPE')),
                                'forwardPE': self._raw_val(summary.get('forwardPE')),
                                'trailingEps': self._raw_val(stats.get('trailingEps')),
                                'forwardEps': self._raw_val(stats.get('forwardEps')),
                                'sharesOutstanding': self._raw_val(stats.get('sharesOutstanding')),
                                'totalRevenue': self._raw_val(financial.get('totalRevenue')),
                                'netIncomeToCommon': self._raw_val(financial.get('netIncomeToCommon', stats.get('netIncomeToCommon'))),
                                'totalDebt': self._raw_val(financial.get('totalDebt')),
                                'totalCash': self._raw_val(financial.get('totalCash')),
                                'revenueGrowth': self._raw_val(financial.get('revenueGrowth')),
                                'earningsGrowth': self._raw_val(financial.get('earningsGrowth')),
                                'profitMargins': self._raw_val(financial.get('profitMargins')),
                                'grossMargins': self._raw_val(financial.get('grossMargins')),
                                'operatingMargins': self._raw_val(financial.get('operatingMargins')),
                                'sector': self._raw_val(price_data.get('sector')) or '',
                                'targetMeanPrice': self._raw_val(financial.get('targetMeanPrice')),
                                'targetMedianPrice': self._raw_val(financial.get('targetMedianPrice')),
                                'targetHighPrice': self._raw_val(financial.get('targetHighPrice')),
                                'targetLowPrice': self._raw_val(financial.get('targetLowPrice')),
                                'numberOfAnalystOpinions': self._raw_val(financial.get('numberOfAnalystOpinions')),
                                'recommendationKey': self._raw_val(financial.get('recommendationKey')),
                            }
                            # Only add non-None values
                            for k, v in enriched.items():
                                if v is not None:
                                    self._info[k] = v
                            print(f"_FallbackTicker({self.ticker}): enriched with v10 data")
                    except Exception as e2:
                        print(f"_FallbackTicker({self.ticker}): v10 enrichment failed ({e2}), using v8 only")

                except Exception as e:
                    print(f"_FallbackTicker({self.ticker}): error: {e}")
                    self._info = {'error': str(e)}
            return self._info
        def _raw_val(self, v):
            if v is None: return None
            if isinstance(v, dict): return v.get('raw')
            return v
        def history(self, **kwargs):
            return _raw_yahoo_history(self.ticker, **kwargs)
        @property
        def income_stmt(self): return pd.DataFrame()
        @property
        def balance_sheet(self): return pd.DataFrame()
        @property
        def cashflow(self): return pd.DataFrame()
        @property
        def revenue_estimate(self): return None
        @property
        def earnings_estimate(self): return None
        @property
        def earnings_history(self): return None
        @property
        def growth_estimates(self): return None
        @property
        def analyst_price_targets(self): return None
        @property
        def news(self): return []
    return _FallbackTicker(symbol)

def yf_safe_history(ticker_obj, **kwargs):
    """Safely get history with retry for flaky connections."""
    for attempt in range(3):
        try:
            h = ticker_obj.history(**kwargs)
            if h is not None and not h.empty:
                return h
        except Exception:
            _time.sleep(1 + attempt)
    return pd.DataFrame()

app = Flask(__name__)
CORS(app)

# ─── Server-side cache with TTL ───
import time as _time

_server_cache = {}
_CACHE_TTL = int(os.environ.get('CACHE_TTL', 600))  # 10 min default, configurable

def _get_cached(key):
    entry = _server_cache.get(key)
    if entry and (_time.time() - entry['ts']) < _CACHE_TTL:
        return entry['data']
    return None

def _set_cached(key, data):
    _server_cache[key] = {'data': data, 'ts': _time.time()}

def _parse_years_param():
    """Parse ?years= query param, default 10, clamp to allowed values."""
    try:
        y = int(request.args.get('years', 10))
    except (ValueError, TypeError):
        y = 10
    return y if y in (5, 10, 20) else 10

# ─── Ticker Aliases (Hebrew/English names → Yahoo Finance symbols) ───
TICKER_ALIASES = {
    # US common aliases
    'ADOBE': 'ADBE',
    'GOOGLE': 'GOOGL',
    'FACEBOOK': 'META',

    # TASE Indices
    'TA90': 'TA90.TA',
    'TA125': 'TA125.TA',
    'TA35': 'TA35.TA',

    # Banks
    'LEUMI': 'LUMI.TA', 'לאומי': 'LUMI.TA', 'בנקלאומי': 'LUMI.TA',
    'POALIM': 'POLI.TA', 'HAPOALIM': 'POLI.TA', 'הפועלים': 'POLI.TA', 'בנקהפועלים': 'POLI.TA',
    'DISCOUNT': 'DSCT.TA', 'דיסקונט': 'DSCT.TA', 'בנקדיסקונט': 'DSCT.TA',
    'MIZRAHI': 'MZTF.TA', 'מזרחי': 'MZTF.TA', 'מזרחיטפחות': 'MZTF.TA',
    'FIBI': 'FIBIH.TA', 'בינלאומי': 'FIBIH.TA', 'הבינלאומי': 'FIBIH.TA',

    # Tech & Large Cap
    'NICE_IL': 'NICE.TA', 'נייס': 'NICE.TA',
    'CHECKPNT': 'CHKP.TA', 'צקפוינט': 'CHKP.TA', 'CHECKPOINT': 'CHKP.TA',
    'ELBIT': 'ESLT.TA', 'אלביט': 'ESLT.TA',
    'TEVA_IL': 'TEVA.TA', 'טבע': 'TEVA.TA',
    'ICL_IL': 'ICL.TA',
    'TOWER': 'TSEM.TA', 'טאואר': 'TSEM.TA',
    'SAPIENS': 'SPNS.TA',

    # Telecom
    'BEZEQ': 'BEZQ.TA', 'בזק': 'BEZQ.TA',
    'CELLCOM': 'CEL.TA', 'סלקום': 'CEL.TA',
    'PARTNER': 'PTNR.TA', 'פרטנר': 'PTNR.TA',

    # Real Estate
    'AZRIELI': 'AZRG.TA', 'עזריאלי': 'AZRG.TA',
    'SHIKUN': 'SKBN.TA', 'שיכון': 'SKBN.TA', 'שיכוןובינוי': 'SKBN.TA',
    'AMOT': 'AMOT.TA', 'אמות': 'AMOT.TA',
    'GAZIT': 'GZT.TA', 'גזית': 'GZT.TA',
    'MELISRON': 'MLSR.TA', 'מליסרון': 'MLSR.TA',
    'AIRPORT': 'ARPT.TA', 'אירפורט': 'ARPT.TA',
    'BIGLAIN': 'BIG.TA', 'ביג': 'BIG.TA',

    # Energy & Resources
    'DELEK': 'DLEKG.TA', 'דלק': 'DLEKG.TA',
    'ENLIGHT': 'ENLT.TA', 'אנלייט': 'ENLT.TA',
    'ORMAT_IL': 'ORA.TA', 'אורמת': 'ORA.TA',
    'ENERGEAN': 'EOAN.TA', 'אנרג׳יאן': 'EOAN.TA',
    'RATIO': 'RATI.TA', 'רציו': 'RATI.TA',
    'ISRAMCO': 'ISRA.TA',

    # Insurance & Finance
    'HAREL': 'HARL.TA', 'הראל': 'HARL.TA',
    'MIGDAL': 'MGDL.TA', 'מגדל': 'MGDL.TA',
    'PHOENIX': 'PHOE.TA', 'הפניקס': 'PHOE.TA',
    'CLAL': 'CLIS.TA', 'כלל': 'CLIS.TA',
    'IDI': 'IDII.TA',

    # Food & Consumer
    'STRAUSS': 'STRS.TA', 'שטראוס': 'STRS.TA',
    'OSEM': 'OSEM.TA', 'אוסם': 'OSEM.TA',
    'SHUFERSAL': 'SAE.TA', 'שופרסל': 'SAE.TA',
    'FOX': 'FOX.TA', 'פוקס': 'FOX.TA',
    'CASTRO': 'CSTR.TA', 'קסטרו': 'CSTR.TA',
    'RAMI': 'RMLI.TA', 'רמי': 'RMLI.TA',

    # Industrials
    'ELCO': 'ELCO.TA', 'אלקו': 'ELCO.TA',
    'KOOR': 'KOR.TA',
    'IDB': 'IDBH.TA',
    'ISRAEL_CORP': 'ILCO.TA', 'כיל': 'ILCO.TA',

    # Pharma & Biotech
    'PERRIGO_IL': 'PRGO.TA',
    'COMPUGEN': 'CGEN.TA',
    'CAMTEK': 'CAMT.TA', 'קמטק': 'CAMT.TA',
    'NOVA': 'NVMI.TA',
}

def resolve_ticker(ticker):
    """Resolve ticker aliases. Supports Hebrew names, English names, and direct symbols."""
    upper = ticker.upper().strip()

    # Direct alias lookup
    if upper in TICKER_ALIASES:
        return TICKER_ALIASES[upper]

    # Hebrew lookup (case doesn't apply but strip spaces)
    cleaned = ticker.strip().replace(' ', '')
    if cleaned in TICKER_ALIASES:
        return TICKER_ALIASES[cleaned]

    return upper

# ─── Helpers ───

def safe(info, key, default=None):
    v = info.get(key, default)
    return default if v is None else v

def to_serializable(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (datetime,)): return obj.isoformat()
    return obj

def clean_dict(d):
    return {k: to_serializable(v) for k, v in d.items()}

def sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None for JSON serialization."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

def compute_rsi(prices, period=14):
    """Compute RSI using Wilder's smoothing — matches TradingView/Bloomberg."""
    if len(prices) < period + 1:
        return [None] * len(prices)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Build RSI series using Wilder's exponential smoothing
    rsi_values = []

    # Seed with SMA of first 'period' values
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))

    # Apply Wilder's smoothing for each subsequent value
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

    # Pad front with None (period values needed for seed + 1 for diff)
    pad = [None] * (len(prices) - len(rsi_values))
    return pad + rsi_values

def compute_ma(prices, period=200):
    if len(prices) < period:
        return [None] * len(prices)
    ma = np.convolve(prices, np.ones(period)/period, mode='valid').tolist()
    pad = [None] * (period - 1)
    return pad + ma

def get_historical_pe(ticker_obj, info, years=5):
    """Calculate historical P/E using price history and actual historical EPS."""
    try:
        hist = ticker_obj.history(period=f"{years}y", interval="1mo")
        if hist.empty or 'Close' not in hist.columns:
            return [], []

        eps_ttm_current = safe(info, 'trailingEps', 0)
        if not eps_ttm_current or eps_ttm_current <= 0:
            return [], []

        dates = [d.strftime('%Y-%m') for d in hist.index]
        prices = hist['Close'].tolist()

        # Build TTM EPS timeline from earnings_dates (best source: ~6 years of quarterly EPS)
        ttm_timeline = {}  # ym -> ttm_eps
        try:
            ed = ticker_obj.earnings_dates
            if ed is not None and not ed.empty and 'Reported EPS' in ed.columns:
                eps_entries = []
                for idx, row in ed.iterrows():
                    val = row.get('Reported EPS')
                    if val is not None and not np.isnan(val) and val > 0:
                        eps_entries.append((idx.to_pydatetime().strftime('%Y-%m'), float(val)))
                eps_entries.sort()
                # Build TTM (rolling 4-quarter sum)
                for i in range(3, len(eps_entries)):
                    ttm = sum(e[1] for e in eps_entries[i-3:i+1])
                    ttm_timeline[eps_entries[i][0]] = ttm
        except:
            pass

        # Supplement with annual financials (Diluted EPS)
        annual_eps = {}
        try:
            fin = ticker_obj.financials
            if fin is not None and not fin.empty and 'Diluted EPS' in fin.index:
                for col in fin.columns:
                    yr = col.year
                    mo = f'{col.month:02d}'
                    annual_eps[f'{yr}-{mo}'] = float(fin.loc['Diluted EPS', col])
        except:
            pass

        # Merge: TTM takes priority, fill gaps with annual
        for ym, eps in annual_eps.items():
            if ym not in ttm_timeline:
                ttm_timeline[ym] = eps

        if not ttm_timeline:
            # Fallback to old method (current EPS for all dates)
            pe_values = [round(p / eps_ttm_current, 2) if p and p > 0 else None for p in prices]
            return dates, pe_values

        sorted_ttm = sorted(ttm_timeline.keys())
        earliest_eps_date = sorted_ttm[0]

        # For dates within TTM range, interpolate; for dates before, extrapolate
        # Compute CAGR from available data for backward extrapolation
        cagr = 0.10  # default 10% if can't compute
        if len(sorted_ttm) >= 2:
            first_val = ttm_timeline[sorted_ttm[0]]
            last_val = ttm_timeline[sorted_ttm[-1]]
            # Parse year difference
            y1 = int(sorted_ttm[0][:4]) + int(sorted_ttm[0][5:]) / 12
            y2 = int(sorted_ttm[-1][:4]) + int(sorted_ttm[-1][5:]) / 12
            span = y2 - y1
            if span > 0 and first_val > 0 and last_val > 0:
                cagr = (last_val / first_val) ** (1 / span) - 1
                cagr = max(min(cagr, 0.30), 0.02)  # clamp 2-30%

        def get_eps_for_date(ym):
            if ym in ttm_timeline:
                return ttm_timeline[ym]

            # Find surrounding EPS dates
            lower = [k for k in sorted_ttm if k <= ym]
            upper = [k for k in sorted_ttm if k > ym]

            if lower and upper:
                # Interpolate
                lk, uk = lower[-1], upper[0]
                lv, uv = ttm_timeline[lk], ttm_timeline[uk]
                yl = int(lk[:4]) + int(lk[5:]) / 12
                yu = int(uk[:4]) + int(uk[5:]) / 12
                yc = int(ym[:4]) + int(ym[5:]) / 12
                frac = (yc - yl) / (yu - yl) if (yu - yl) > 0 else 0
                return lv + frac * (uv - lv)
            elif lower:
                # After last known — use current TTM EPS
                return eps_ttm_current
            else:
                # Before earliest known — extrapolate backward with CAGR
                earliest_val = ttm_timeline[sorted_ttm[0]]
                ye = int(sorted_ttm[0][:4]) + int(sorted_ttm[0][5:]) / 12
                yc = int(ym[:4]) + int(ym[5:]) / 12
                years_back = ye - yc
                return earliest_val / ((1 + cagr) ** years_back)

        pe_values = []
        for dt, price in zip(dates, prices):
            eps = get_eps_for_date(dt)
            if eps and eps > 0.01 and price and price > 0:
                pe = price / eps
                pe_values.append(round(pe, 2) if 0 < pe < 500 else None)
            else:
                pe_values.append(None)

        return dates, pe_values
    except:
        return [], []


# ─── API Routes ───

@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/health')
def health():
    # Quick live test of v8 chart API
    v8_ok = False
    try:
        r = http_requests.get(
            'https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=1d',
            headers=_YF_HEADERS, timeout=10)
        v8_ok = r.status_code == 200 and 'chart' in r.text
    except Exception:
        pass
    return jsonify({
        'status': 'ok',
        'yfinance_works': _YF_WORKS,
        'fallback_mode': not _YF_WORKS,
        'yahoo_crumb_ok': _yahoo_crumb is not None and len(_yahoo_crumb or '') < 100,
        'v8_chart_api_ok': v8_ok,
    })


@app.route('/api/search/<query>')
def search_ticker(query):
    """Search for tickers — supports Hebrew names and Yahoo Finance search."""
    query = query.strip()
    results = []

    # 1. Check local aliases first
    for alias, symbol in TICKER_ALIASES.items():
        if query.upper() in alias.upper() or query in alias:
            try:
                t = yf_ticker(symbol)
                info = t.info
                name = info.get('shortName', info.get('longName', alias))
                price = info.get('currentPrice', info.get('regularMarketPrice'))
                results.append({
                    'symbol': symbol,
                    'alias': alias,
                    'name': name,
                    'price': price,
                    'exchange': 'TASE' if '.TA' in symbol else 'US',
                })
            except Exception:
                results.append({'symbol': symbol, 'alias': alias, 'name': alias, 'exchange': 'TASE' if '.TA' in symbol else 'US'})
        if len(results) >= 8:
            break

    # 2. Yahoo Finance search
    if len(results) < 5:
        try:
            import requests as rq
            r = rq.get(
                f'https://query2.finance.yahoo.com/v1/finance/search?q={query}',
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=5,
            )
            data = r.json()
            for q in data.get('quotes', [])[:6]:
                sym = q.get('symbol', '')
                if sym and not any(r['symbol'] == sym for r in results):
                    results.append({
                        'symbol': sym,
                        'name': q.get('shortname', q.get('longname', sym)),
                        'exchange': q.get('exchange', ''),
                        'type': q.get('quoteType', ''),
                    })
        except Exception:
            pass

    return jsonify(results[:10])


@app.route('/api/stock/<ticker>')
def get_stock(ticker):
    """Get comprehensive stock data."""
    years = _parse_years_param()
    cache_key = f'stock_{ticker.upper().strip()}_{years}y'
    cached = _get_cached(cache_key)
    if cached:
        return jsonify(cached)
    try:
        original_ticker = ticker.upper().strip()
        resolved = resolve_ticker(original_ticker)
        t = yf_ticker(resolved)
        info = t.info

        if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None:
            return jsonify({'error': f'Ticker "{ticker}" not found'}), 404

        # Currency detection for Israeli stocks
        trade_currency = info.get('currency', 'USD')
        financial_currency = info.get('financialCurrency', 'USD')
        is_ils = trade_currency == 'ILA' or '.TA' in resolved
        agorot_divisor = 100 if trade_currency == 'ILA' else 1  # ILA = agorot, divide by 100 for shekels

        price = safe(info, 'currentPrice', safe(info, 'regularMarketPrice', 0))
        prev_close = safe(info, 'previousClose', price)
        change = price - prev_close if price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        # Historical data — daily for ALL charts
        hist5y = yf_safe_history(t, period=f"{years}y", interval="1d")
        if hist5y.empty or 'Close' not in hist5y.columns:
            dates, closes, volumes = [], [], []
        else:
            dates = [d.strftime('%Y-%m-%d') for d in hist5y.index]
            closes = hist5y['Close'].tolist()
            volumes = hist5y['Volume'].tolist() if 'Volume' in hist5y.columns else []

        # Technical indicators — RSI(14) standard, MA(200)
        if closes:
            rsi_values = compute_rsi(np.array(closes), period=14)
            ma200 = compute_ma(closes, 200)
        else:
            rsi_values = []
            ma200 = []

        current_rsi = rsi_values[-1] if rsi_values and rsi_values[-1] is not None else 50

        # Historical P/E
        pe_dates, pe_values = get_historical_pe(t, info, years=years)
        current_pe = safe(info, 'trailingPE')
        forward_pe = safe(info, 'forwardPE')

        # P/E percentile
        valid_pes = [p for p in pe_values if p is not None and 0 < p < 200]
        pe_percentile = None
        pe_verdict = 'N/A'
        if valid_pes and current_pe:
            pe_percentile = sum(1 for p in valid_pes if p < current_pe) / len(valid_pes) * 100
            if pe_percentile > 75:
                pe_verdict = 'Expensive'
            elif pe_percentile < 25:
                pe_verdict = 'Cheap'
            else:
                pe_verdict = 'Fair'

        # Signal logic
        pe_avg = np.mean(valid_pes) if valid_pes else None
        signal = 'Neutral'
        signal_color = 'yellow'
        if current_rsi and pe_avg and current_pe:
            if current_rsi < 30 and current_pe < pe_avg:
                signal = 'Strong Buy'
                signal_color = 'green'
            elif current_rsi < 40 and current_pe < pe_avg:
                signal = 'Buy'
                signal_color = 'green'
            elif current_rsi > 70 and current_pe > pe_avg:
                signal = 'Strong Sell'
                signal_color = 'red'
            elif current_rsi > 60 and current_pe > pe_avg:
                signal = 'Sell'
                signal_color = 'red'

        # Fundamentals
        fundamentals = {
            'revenueGrowth': safe(info, 'revenueGrowth'),
            'earningsGrowth': safe(info, 'earningsGrowth'),
            'grossMargins': safe(info, 'grossMargins'),
            'operatingMargins': safe(info, 'operatingMargins'),
            'profitMargins': safe(info, 'profitMargins'),
            'debtToEquity': safe(info, 'debtToEquity'),
            'freeCashflow': safe(info, 'freeCashflow'),
            'returnOnEquity': safe(info, 'returnOnEquity'),
            'totalRevenue': safe(info, 'totalRevenue'),
            'totalDebt': safe(info, 'totalDebt'),
            'totalCash': safe(info, 'totalCash'),
            'ebitda': safe(info, 'ebitda'),
        }

        # PEG ratio
        peg_ratio = safe(info, 'trailingPegRatio')
        if peg_ratio is None:
            # Calculate PEG from forward PE and earnings growth
            eg = safe(info, 'earningsGrowth')
            if forward_pe and eg and eg > 0:
                peg_ratio = round(forward_pe / (eg * 100), 2)

        # Historical Forward P/E and PEG
        # Forward PE: scale trailing PE by current trailing/forward EPS ratio
        # PEG: use implied long-term growth from yfinance's trailingPegRatio
        fwd_pe_dates = []
        fwd_pe_values = []
        peg_hist_dates = []
        peg_hist_values = []
        try:
            # Forward PE ratio: trailing_eps / forward_eps (consistent across time)
            forward_eps_val = safe(info, 'forwardEps')
            trailing_eps_val = safe(info, 'trailingEps')
            fwd_ratio = (trailing_eps_val / forward_eps_val) if (
                forward_eps_val and trailing_eps_val and forward_eps_val > 0
            ) else None

            # PEG implied growth: derived from yfinance's trailingPegRatio
            # This is the analyst consensus long-term growth (5yr expected)
            implied_growth_pct = None
            if peg_ratio and peg_ratio > 0 and current_pe and current_pe > 0:
                implied_growth_pct = current_pe / peg_ratio  # as percentage

            for dt, pe_val in zip(pe_dates, pe_values):
                if pe_val is None or pe_val <= 0:
                    continue

                # Forward PE = trailing PE * (trailing EPS / forward EPS)
                if fwd_ratio:
                    fwd_pe = pe_val * fwd_ratio
                    if 2 < fwd_pe < 500:
                        fwd_pe_dates.append(dt)
                        fwd_pe_values.append(round(fwd_pe, 2))

                # PEG = trailing PE / implied long-term growth %
                if implied_growth_pct and implied_growth_pct > 0:
                    peg_val = round(pe_val / implied_growth_pct, 2)
                    if 0.1 < peg_val < 10:
                        peg_hist_dates.append(dt)
                        peg_hist_values.append(peg_val)
        except Exception as e:
            print(f"Forward PE/PEG history error: {e}")

        # Valuation metrics
        valuation = {
            'trailingPE': current_pe,
            'forwardPE': forward_pe,
            'priceToSales': safe(info, 'priceToSalesTrailing12Months'),
            'priceToBook': safe(info, 'priceToBook'),
            'evToEbitda': safe(info, 'enterpriseToEbitda'),
            'pegRatio': peg_ratio,
            'pePercentile': pe_percentile,
            'peVerdict': pe_verdict,
            'peAvgHistorical': pe_avg,
        }

        # FCF history for DCF
        fcf_history = []
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                for col in cf.columns:
                    try:
                        if "Free Cash Flow" in cf.index:
                            val = cf.loc["Free Cash Flow", col]
                            if val is not None and not np.isnan(val):
                                fcf_history.append({'year': col.year, 'fcf': float(val)})
                        else:
                            ocf = cf.loc.get("Operating Cash Flow", {}).get(col)
                            capex = cf.loc.get("Capital Expenditure", {}).get(col)
                            if ocf is not None and capex is not None:
                                fcf_history.append({'year': col.year, 'fcf': float(ocf + capex)})
                    except:
                        pass
        except:
            pass
        fcf_history.sort(key=lambda x: x['year'])

        # DCF inputs
        dcf_inputs = {
            'fcfHistory': fcf_history,
            'sharesOutstanding': safe(info, 'sharesOutstanding', 0),
            'totalDebt': safe(info, 'totalDebt', 0),
            'totalCash': safe(info, 'totalCash', 0),
            'earningsGrowth': safe(info, 'earningsGrowth', 0.1),
            'revenueGrowth': safe(info, 'revenueGrowth', 0.1),
            'beta': safe(info, 'beta', 1.0),
        }

        # News (from yfinance - new nested format)
        news = []
        try:
            yf_news = t.news
            if yf_news:
                for article in yf_news[:10]:
                    # yfinance 0.2.x uses nested 'content' structure
                    content = article.get('content', article)
                    title = content.get('title', article.get('title', ''))
                    publisher = ''
                    link = ''
                    pub_date = ''

                    # Extract provider
                    provider = content.get('provider', {})
                    if isinstance(provider, dict):
                        publisher = provider.get('displayName', '')

                    # Extract URL
                    canonical = content.get('canonicalUrl', {})
                    if isinstance(canonical, dict):
                        link = canonical.get('url', '')
                    if not link:
                        click = content.get('clickThroughUrl', {})
                        if isinstance(click, dict):
                            link = click.get('url', '')
                    if not link:
                        link = article.get('link', '')

                    # Extract date
                    pub_date = content.get('pubDate', content.get('displayTime', article.get('providerPublishTime', '')))

                    if not title:
                        continue

                    # Simple sentiment
                    positive_words = ['surge', 'gain', 'rise', 'up', 'beat', 'strong', 'growth', 'profit', 'bull', 'buy', 'upgrade', 'outperform', 'positive', 'record', 'high', 'soar', 'rally']
                    negative_words = ['fall', 'drop', 'decline', 'down', 'miss', 'weak', 'loss', 'bear', 'sell', 'downgrade', 'underperform', 'negative', 'low', 'crash', 'fear', 'plunge', 'wipe', 'sell-off']
                    title_lower = title.lower()
                    pos_count = sum(1 for w in positive_words if w in title_lower)
                    neg_count = sum(1 for w in negative_words if w in title_lower)
                    if pos_count > neg_count:
                        sentiment = 'positive'
                    elif neg_count > pos_count:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'

                    news.append({
                        'title': title,
                        'publisher': publisher,
                        'link': link,
                        'timestamp': pub_date,
                        'sentiment': sentiment,
                    })
        except:
            pass

        result_data = {
            'ticker': original_ticker,
            'resolvedTicker': resolved,
            'currency': '₪' if is_ils else '$',
            'currencyCode': 'ILS' if is_ils else 'USD',
            'financialCurrency': financial_currency,
            'isILS': is_ils,
            'agorotDivisor': agorot_divisor,
            'name': safe(info, 'longName', ticker.upper()),
            'sector': safe(info, 'sector', 'N/A'),
            'industry': safe(info, 'industry', 'N/A'),
            'price': price,
            'previousClose': prev_close,
            'change': round(change, 2),
            'changePct': round(change_pct, 2),
            'marketCap': safe(info, 'marketCap', 0),
            'volume': safe(info, 'volume', 0),
            'fiftyTwoWeekHigh': safe(info, 'fiftyTwoWeekHigh'),
            'fiftyTwoWeekLow': safe(info, 'fiftyTwoWeekLow'),
            'dividendYield': safe(info, 'dividendYield'),
            'chart': {
                'dates': dates,
                'closes': closes,
                'volumes': volumes,
                'rsi': rsi_values,
                'ma200': ma200,
            },
            'historicalPE': {
                'dates': pe_dates,
                'values': pe_values,
            },
            'historicalForwardPE': {
                'dates': fwd_pe_dates,
                'values': fwd_pe_values,
            },
            'historicalPEG': {
                'dates': peg_hist_dates,
                'values': peg_hist_values,
            },
            'valuation': clean_dict(valuation),
            'fundamentals': clean_dict(fundamentals),
            'dcfInputs': clean_dict(dcf_inputs),
            'signal': signal,
            'signalColor': signal_color,
            'currentRSI': round(current_rsi, 1) if current_rsi else None,
            'news': news,
            '_fetchedAt': datetime.now().isoformat(),
        }
        result_data = sanitize_for_json(result_data)
        _set_cached(cache_key, result_data)
        return jsonify(result_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/dcf', methods=['POST'])
def calculate_dcf():
    """Calculate DCF valuation with user parameters."""
    try:
        data = request.json
        last_fcf = data.get('lastFCF', 0)
        growth_rate_1_5 = data.get('growthRate', 0.10)
        growth_rate_6_10 = growth_rate_1_5 / 2
        terminal_growth = data.get('terminalGrowth', 0.025)
        wacc = data.get('wacc', 0.10)
        shares = data.get('shares', 1)
        debt = data.get('debt', 0)
        cash = data.get('cash', 0)

        if last_fcf <= 0 or wacc <= terminal_growth:
            return jsonify({'error': 'Invalid inputs'}), 400

        # Project FCFs
        projected = []
        for yr in range(1, 11):
            g = growth_rate_1_5 if yr <= 5 else growth_rate_6_10
            fcf = last_fcf * (1 + g) ** yr if yr <= 5 else projected[4] * (1 + growth_rate_6_10) ** (yr - 5)
            pv = fcf / (1 + wacc) ** yr
            projected.append(fcf)

        pv_fcfs = [projected[i] / (1 + wacc) ** (i+1) for i in range(10)]
        sum_pv = sum(pv_fcfs)

        # Terminal value
        tv = projected[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_tv = tv / (1 + wacc) ** 10

        ev = sum_pv + pv_tv
        equity = ev - debt + cash
        intrinsic = equity / shares if shares > 0 else 0

        # Sensitivity table
        growth_rates = [growth_rate_1_5 * m for m in [0.5, 0.75, 1.0, 1.25, 1.5]]
        discount_rates = [wacc + d for d in [-0.02, -0.01, 0, 0.01, 0.02]]

        sensitivity = []
        for gr in growth_rates:
            row = []
            for dr in discount_rates:
                if dr <= terminal_growth:
                    row.append(None)
                    continue
                proj = []
                for yr in range(1, 11):
                    g = gr if yr <= 5 else gr / 2
                    f = last_fcf * (1 + g) ** yr if yr <= 5 else proj[4] * (1 + gr/2) ** (yr - 5)
                    proj.append(f)
                pvs = [proj[i] / (1 + dr) ** (i+1) for i in range(10)]
                t_v = proj[-1] * (1 + terminal_growth) / (dr - terminal_growth)
                pv_t = t_v / (1 + dr) ** 10
                eq = sum(pvs) + pv_t - debt + cash
                row.append(round(eq / shares, 2) if shares > 0 else 0)
            sensitivity.append(row)

        return jsonify({
            'intrinsicValue': round(intrinsic, 2),
            'projectedFCFs': [round(f, 0) for f in projected],
            'pvFCFs': [round(f, 0) for f in pv_fcfs],
            'terminalValue': round(tv, 0),
            'pvTerminal': round(pv_tv, 0),
            'enterpriseValue': round(ev, 0),
            'equityValue': round(equity, 0),
            'sensitivity': {
                'growthRates': [round(g, 4) for g in growth_rates],
                'discountRates': [round(d, 4) for d in discount_rates],
                'values': sensitivity,
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Macro Dashboard Endpoint ───

@app.route('/api/macro')
def get_macro_data():
    """Get macro market overview: Fear & Greed, VIX, USD/ILS, S&P500 metrics."""
    years = _parse_years_param()
    cache_key = f'macro_dashboard_{years}y'
    cached = _get_cached(cache_key)
    if cached:
        return jsonify(cached)

    from concurrent.futures import ThreadPoolExecutor
    result = {}

    def _fetch_fear_greed():
        return _fetch_fg_inner()

    def _fetch_vix():
        return _fetch_vix_inner(years)

    def _fetch_usdils():
        return _fetch_ils_inner(years)

    def _fetch_sp500():
        return _fetch_sp_inner(years)

    def _fetch_ta35():
        return _fetch_ta35_inner()

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {
            ex.submit(_fetch_fear_greed): 'fearGreed',
            ex.submit(_fetch_vix): 'vix',
            ex.submit(_fetch_usdils): 'usdIls',
            ex.submit(_fetch_sp500): 'sp500',
            ex.submit(_fetch_ta35): 'ta35',
        }
        for future in futures:
            key = futures[future]
            try:
                result[key] = future.result(timeout=45)
            except Exception as e:
                result[key] = {'error': str(e)}

    result['_fetchedAt'] = datetime.now().isoformat()
    result['years'] = years
    result = sanitize_for_json(result)
    _set_cached(cache_key, result)
    return jsonify(result)


def _fetch_fg_inner():
    try:
        r = http_requests.get(
            'https://production.dataviz.cnn.io/index/fearandgreed/graphdata',
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'},
            timeout=10,
        )
        fg = r.json().get('fear_and_greed', {})
        return {
            'score': round(fg.get('score', 0), 1),
            'rating': fg.get('rating', 'N/A'),
            'previousClose': round(fg.get('previous_close', 0), 1),
            'oneWeekAgo': round(fg.get('previous_1_week', 0), 1),
            'oneMonthAgo': round(fg.get('previous_1_month', 0), 1),
            'oneYearAgo': round(fg.get('previous_1_year', 0), 1),
        }
    except Exception as e:
        return {'score': None, 'rating': 'N/A', 'error': str(e)}


def _fetch_vix_inner(years=10):
    try:
        vix = yf_ticker('^VIX')
        vix_hist = yf_safe_history(vix, period=f'{years}y', interval='1mo')
        if vix_hist.empty or 'Close' not in vix_hist.columns:
            return {'current': None, 'error': 'No VIX data returned'}
        vix_price = float(vix_hist['Close'].iloc[-1])
        vix_dates = [d.strftime('%Y-%m') for d in vix_hist.index]
        vix_closes = [round(float(c), 2) for c in vix_hist['Close'].tolist()]
        vix_avg = round(float(np.mean(vix_closes)), 2)
        return {
            'current': round(vix_price, 2) if vix_price else None,
            'dates': vix_dates, 'closes': vix_closes,
            'avg10y': vix_avg, 'min10y': round(min(vix_closes), 2), 'max10y': round(max(vix_closes), 2),
        }
    except Exception as e:
        return {'current': None, 'error': str(e)}


def _fetch_ils_inner(years=10):
    try:
        usdils = yf_ticker('ILS=X')
        ils_hist = yf_safe_history(usdils, period=f'{years}y', interval='1mo')
        if ils_hist.empty or 'Close' not in ils_hist.columns:
            return {'current': None, 'error': 'No ILS data returned'}
        ils_price = float(ils_hist['Close'].iloc[-1])
        ils_dates = [d.strftime('%Y-%m') for d in ils_hist.index]
        ils_closes = [round(float(c), 4) for c in ils_hist['Close'].tolist()]
        return {
            'current': round(ils_price, 4) if ils_price else None,
            'dates': ils_dates, 'closes': ils_closes,
            'avg10y': round(float(np.mean(ils_closes)), 4),
            'min10y': round(min(ils_closes), 4), 'max10y': round(max(ils_closes), 4),
        }
    except Exception as e:
        return {'current': None, 'error': str(e)}


def _fetch_sp_inner(years=10):
    try:
        sp = yf_ticker('^GSPC')
        sp_hist = yf_safe_history(sp, period=f'{years}y', interval='1mo')
        if sp_hist.empty or 'Close' not in sp_hist.columns:
            return {'price': None, 'error': 'No S&P 500 data returned'}
        sp_price = float(sp_hist['Close'].iloc[-1])
        sp_dates = [d.strftime('%Y-%m') for d in sp_hist.index]
        sp_closes = [round(float(c), 2) for c in sp_hist['Close'].tolist()]

        # RSI(14) + MA200 from daily — use 2y history for accuracy
        sp_daily_long = yf_safe_history(sp, period='2y', interval='1d')
        sp_rsi = None
        ma200 = None
        if not sp_daily_long.empty and 'Close' in sp_daily_long.columns:
            closes_arr = sp_daily_long['Close'].values
            # RSI using Wilder's smoothing
            rsi_series = compute_rsi(closes_arr, period=14)
            if rsi_series and rsi_series[-1] is not None:
                sp_rsi = round(rsi_series[-1], 1)
            # MA200
            if len(sp_daily_long) >= 200:
                ma200_val = sp_daily_long['Close'].rolling(200).mean().iloc[-1]
                if not np.isnan(ma200_val):
                    ma200 = round(float(ma200_val), 2)

        # Change
        if not sp_daily_long.empty and 'Close' in sp_daily_long.columns and len(sp_daily_long) > 1:
            prev = float(sp_daily_long['Close'].iloc[-2])
            change_pct = round((float(sp_daily_long['Close'].iloc[-1]) - prev) / prev * 100, 2)
        else:
            change_pct = 0

        # S&P 500 P/E from multpl.com (10-year history)
        sp_pe = None
        sp_fwd_pe = None
        pe_history = []
        try:
            from bs4 import BeautifulSoup
            r = http_requests.get('https://www.multpl.com/s-p-500-pe-ratio/table/by-month',
                                  headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'},
                                  timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            table = soup.find('table', {'id': 'datatable'})
            if table:
                for row in table.find_all('tr')[1:years * 13]:  # monthly data
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        date_str = cols[0].text.strip()
                        pe_str = cols[1].text.strip().replace('†\n', '').strip()
                        try:
                            pe_val = float(pe_str)
                            pe_history.append({'date': date_str, 'pe': pe_val})
                        except:
                            pass
            if pe_history:
                sp_pe = pe_history[0]['pe']

        except Exception:
            pass

        # Forward PE — calculate from earnings data + growth projection
        # since multpl.com forward PE page is no longer available
        try:
            from bs4 import BeautifulSoup as _BS_fwd
            from datetime import datetime as _dt_fwd
            r_earn = http_requests.get('https://www.multpl.com/s-p-500-earnings/table/by-month',
                                       headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'},
                                       timeout=10)
            if r_earn.status_code == 200:
                soup_earn = _BS_fwd(r_earn.text, 'html.parser')
                tbl_earn = soup_earn.find('table', {'id': 'datatable'})
                if tbl_earn:
                    earn_rows = []
                    for row in tbl_earn.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        if len(cols) >= 2:
                            try:
                                d = _dt_fwd.strptime(cols[0].text.strip().replace(',', ''), '%b %d %Y')
                                val = float(cols[1].text.strip().replace('†\n', '').replace('$', '').strip())
                                earn_rows.append((d, val))
                            except:
                                pass
                    # Get latest earnings and same month 1 year ago for YoY growth
                    if len(earn_rows) >= 13:
                        latest_eps = earn_rows[0][1]
                        yr_ago_eps = earn_rows[12][1]  # ~12 months back
                        if yr_ago_eps > 0 and latest_eps > 0:
                            yoy_growth = (latest_eps - yr_ago_eps) / yr_ago_eps
                            # Cap growth projection at 20% to be conservative
                            proj_growth = min(max(yoy_growth, 0.02), 0.20)
                            fwd_eps = latest_eps * (1 + proj_growth)
                            if sp_price and fwd_eps > 0:
                                sp_fwd_pe = round(sp_price / fwd_eps, 2)
        except Exception as e:
            print(f"Forward PE calculation failed: {e}")

        pe_dates = [p['date'] for p in pe_history]
        pe_values = [p['pe'] for p in pe_history]

        # PEG Ratio = Forward PE / LTEG (Long-Term Expected Growth)
        # Yardeni standard: Forward P/E divided by long-term expected EPS growth
        # We estimate forward PE from trailing PE + growth, then divide by LTEG
        peg_current = None
        peg_dates = []
        peg_values = []
        fwd_pe_dates = []
        fwd_pe_values = []
        try:
            from bs4 import BeautifulSoup as _BS
            from datetime import datetime as _dt
            re_earn = http_requests.get('https://www.multpl.com/s-p-500-earnings/table/by-month',
                                   headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'},
                                   timeout=10)
            earn_dict = {}
            if re_earn.status_code == 200:
                soup_e = _BS(re_earn.text, 'html.parser')
                table_e = soup_e.find('table', {'id': 'datatable'})
                if table_e:
                    for row in table_e.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        if len(cols) >= 2:
                            try:
                                d = _dt.strptime(cols[0].text.strip().replace(',', ''), '%b %d %Y')
                                val = float(cols[1].text.strip().replace('†\n', '').replace('$', '').strip())
                                earn_dict[d.strftime('%Y-%m')] = val
                            except:
                                pass

            # Build trailing PE dict from pe_history
            pe_dict = {}
            for p in pe_history:
                try:
                    d = _dt.strptime(p['date'].replace(',', ''), '%b %d %Y')
                    pe_dict[d.strftime('%Y-%m')] = p['pe']
                except:
                    pass

            sorted_months = sorted(pe_dict.keys(), reverse=True)[:years * 13]
            for ym in sorted_months:
                yr, mo = int(ym[:4]), ym[5:]

                # Calculate YoY earnings growth for each of the past 5 years
                growth_rates = []
                for y_offset in range(5):
                    cur_ym = f'{yr - y_offset}-{mo}'
                    prev_ym = f'{yr - y_offset - 1}-{mo}'
                    if cur_ym in earn_dict and prev_ym in earn_dict and earn_dict[prev_ym] > 0:
                        g = (earn_dict[cur_ym] - earn_dict[prev_ym]) / earn_dict[prev_ym] * 100
                        growth_rates.append(g)

                if len(growth_rates) >= 3:
                    # LTEG: clip negative rates to 0, take mean, floor at 5%
                    clipped = [max(g, 0) for g in growth_rates]
                    avg_growth = sum(clipped) / len(clipped)
                    avg_growth = max(avg_growth, 5.0)

                    # Estimate forward PE from trailing PE and projected growth
                    proj_g = min(max(avg_growth / 100, 0.02), 0.20)
                    fwd_pe_ym = pe_dict[ym] / (1 + proj_g)

                    # Collect forward PE history
                    if 5 < fwd_pe_ym < 60:
                        fwd_pe_dates.append(ym)
                        fwd_pe_values.append(round(fwd_pe_ym, 2))

                    peg = round(fwd_pe_ym / avg_growth, 2)
                    if 0.3 < peg < 4.0:
                        peg_dates.append(ym)
                        peg_values.append(peg)

            # For the current month, use the already-calculated sp_fwd_pe if available
            if sp_fwd_pe and peg_values:
                # Recompute current PEG with the actual forward PE
                latest_ym = peg_dates[0] if peg_dates else None
                if latest_ym:
                    yr, mo = int(latest_ym[:4]), latest_ym[5:]
                    growth_rates_cur = []
                    for y_offset in range(5):
                        cur_ym = f'{yr - y_offset}-{mo}'
                        prev_ym = f'{yr - y_offset - 1}-{mo}'
                        if cur_ym in earn_dict and prev_ym in earn_dict and earn_dict[prev_ym] > 0:
                            g = (earn_dict[cur_ym] - earn_dict[prev_ym]) / earn_dict[prev_ym] * 100
                            growth_rates_cur.append(g)
                    if len(growth_rates_cur) >= 3:
                        clipped_cur = [max(g, 0) for g in growth_rates_cur]
                        lteg_cur = max(sum(clipped_cur) / len(clipped_cur), 5.0)
                        peg_current = round(sp_fwd_pe / lteg_cur, 2)
                        peg_values[0] = peg_current
            # Update forward PE history with actual calculated sp_fwd_pe for current month
            if sp_fwd_pe and fwd_pe_dates:
                fwd_pe_values[0] = sp_fwd_pe
            if peg_current is None and peg_values:
                peg_current = peg_values[0]
        except Exception as e:
            print(f"PEG calculation failed: {e}")

        return {
            'price': round(sp_price, 2) if sp_price else None,
            'change_pct': change_pct,
            'rsi': sp_rsi,
            'pe': sp_pe,
            'forwardPE': sp_fwd_pe,
            'peHistory': {'dates': pe_dates, 'values': pe_values},
            'peg': peg_current,
            'pegHistory': {'dates': peg_dates, 'values': peg_values},
            'forwardPEHistory': {'dates': fwd_pe_dates, 'values': fwd_pe_values},
            'ma200': ma200,
            'dates': sp_dates,
            'closes': sp_closes,
        }
    except Exception as e:
        return {'price': None, 'error': str(e)}


def _fetch_ta35_inner():
    try:
        ta35 = yf_ticker('TA35.TA')
        ta35_hist = yf_safe_history(ta35, period='5d', interval='1d')
        if not ta35_hist.empty and 'Close' in ta35_hist.columns and len(ta35_hist) >= 2:
            ta35_price = float(ta35_hist['Close'].iloc[-1])
            ta35_prev = float(ta35_hist['Close'].iloc[-2])
            ta35_change = round((ta35_price - ta35_prev) / ta35_prev * 100, 2)
            return {'price': round(ta35_price, 2), 'change_pct': ta35_change}
        return {'price': None}
    except Exception as e:
        return {'price': None, 'error': str(e)}


# ─── P/E Model Endpoint ───

@app.route('/api/pemodel/<ticker>')
def get_pe_model_data(ticker):
    """Get data for P/E model — uses analyst consensus estimates for conservative projections."""
    try:
        resolved = resolve_ticker(ticker.upper().strip())
        t = yf_ticker(resolved)
        info = t.info

        revenue = info.get('totalRevenue', 0) or 0
        net_income = info.get('netIncomeToCommon', 0) or 0
        shares = info.get('sharesOutstanding', 0) or 0
        price = safe(info, 'currentPrice', safe(info, 'regularMarketPrice', 0))
        eps = info.get('trailingEps', 0) or 0
        pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        forward_eps = info.get('forwardEps')
        market_cap = info.get('marketCap', 0) or 0
        rev_growth = info.get('revenueGrowth')
        profit_margin = info.get('profitMargins')

        # ─── Analyst Estimates (consensus) ───
        analyst = {
            'targetMean': info.get('targetMeanPrice'),
            'targetMedian': info.get('targetMedianPrice'),
            'targetLow': info.get('targetLowPrice'),
            'targetHigh': info.get('targetHighPrice'),
            'numAnalysts': info.get('numberOfAnalystOpinions'),
            'recommendation': info.get('recommendationKey'),
            'recommendationScore': info.get('recommendationMean'),
        }

        # Revenue estimates from analysts
        rev_estimates = {}
        try:
            re = t.revenue_estimate
            if re is not None and not re.empty:
                for idx in re.index:
                    row = re.loc[idx]
                    rev_estimates[idx] = {
                        'avg': float(row['avg']) if 'avg' in row and row['avg'] == row['avg'] else None,
                        'low': float(row['low']) if 'low' in row and row['low'] == row['low'] else None,
                        'high': float(row['high']) if 'high' in row and row['high'] == row['high'] else None,
                        'growth': float(row['growth']) if 'growth' in row and row['growth'] == row['growth'] else None,
                        'numAnalysts': int(row['numberOfAnalysts']) if 'numberOfAnalysts' in row and row['numberOfAnalysts'] == row['numberOfAnalysts'] else None,
                    }
        except Exception:
            pass

        # EPS estimates from analysts
        eps_estimates = {}
        try:
            ee = t.earnings_estimate
            if ee is not None and not ee.empty:
                for idx in ee.index:
                    row = ee.loc[idx]
                    eps_estimates[idx] = {
                        'avg': float(row['avg']) if 'avg' in row and row['avg'] == row['avg'] else None,
                        'low': float(row['low']) if 'low' in row and row['low'] == row['low'] else None,
                        'high': float(row['high']) if 'high' in row and row['high'] == row['high'] else None,
                        'growth': float(row['growth']) if 'growth' in row and row['growth'] == row['growth'] else None,
                        'numAnalysts': int(row['numberOfAnalysts']) if 'numberOfAnalysts' in row and row['numberOfAnalysts'] == row['numberOfAnalysts'] else None,
                    }
        except Exception:
            pass

        # Conservative growth: use LOWER of analyst consensus
        # Year 1: analyst current year growth (0y)
        # Year 2: analyst next year growth (+1y)
        # Year 3+: decelerate toward long-term avg
        analyst_rev_growth_y0 = rev_estimates.get('0y', {}).get('growth')
        analyst_rev_growth_y1 = rev_estimates.get('+1y', {}).get('growth')
        analyst_eps_growth_y0 = eps_estimates.get('0y', {}).get('growth')
        analyst_eps_growth_y1 = eps_estimates.get('+1y', {}).get('growth')

        # Conservative: use the LOWER estimate between revenue and earnings growth
        suggested_growth = None
        if analyst_rev_growth_y1 is not None:
            suggested_growth = analyst_rev_growth_y1  # next year (already more conservative)
        elif analyst_rev_growth_y0 is not None:
            suggested_growth = analyst_rev_growth_y0 * 0.85  # discount current year by 15%
        elif rev_growth is not None:
            suggested_growth = rev_growth * 0.7  # discount trailing by 30%

        # Cap at reasonable levels
        if suggested_growth is not None:
            suggested_growth = min(suggested_growth, 0.30)  # max 30%
            suggested_growth = max(suggested_growth, 0.02)  # min 2%

        # Conservative margin: use forward EPS / forward revenue if available
        suggested_margin = profit_margin
        if forward_eps and rev_estimates.get('0y', {}).get('avg'):
            fwd_net_income = forward_eps * shares
            fwd_revenue = rev_estimates['0y']['avg']
            if fwd_revenue > 0:
                suggested_margin = fwd_net_income / fwd_revenue

        # Conservative P/E: use forward P/E (lower than trailing)
        suggested_pe = forward_pe or pe
        if suggested_pe:
            suggested_pe = min(suggested_pe, pe or suggested_pe)  # use the lower

        # Historical revenue
        hist_revenue = []
        try:
            inc = t.income_stmt
            if inc is not None and not inc.empty:
                if 'Total Revenue' in inc.index:
                    for col in inc.columns:
                        val = inc.loc['Total Revenue', col]
                        if val and not np.isnan(val):
                            hist_revenue.append({'year': col.year, 'revenue': float(val)})
                if 'Net Income' in inc.index:
                    ni = inc.loc['Net Income'].iloc[0]
                    if ni and not np.isnan(ni):
                        net_income = float(ni)
        except Exception:
            pass
        hist_revenue.sort(key=lambda x: x['year'])

        _is_ils = info.get('currency') == 'ILA' or '.TA' in resolved
        _agorot = 100 if info.get('currency') == 'ILA' else 1

        return jsonify(sanitize_for_json({
            'ticker': ticker.upper(),
            'name': safe(info, 'longName', ticker),
            'revenue': revenue,
            'netIncome': net_income,
            'sharesOutstanding': shares,
            'currentPrice': price,
            'eps': eps,
            'forwardEps': forward_eps,
            'trailingPE': pe,
            'forwardPE': forward_pe,
            'marketCap': market_cap,
            'revenueGrowth': rev_growth,
            'profitMargin': profit_margin,
            'histRevenue': hist_revenue,
            'isILS': _is_ils,
            'currencySymbol': '₪' if _is_ils else '$',
            'agorotDivisor': _agorot,
            'financialCurrency': info.get('financialCurrency', 'USD'),
            # Analyst data
            'analyst': analyst,
            'revEstimates': rev_estimates,
            'epsEstimates': eps_estimates,
            # Conservative suggestions
            'suggestedGrowth': round(suggested_growth * 100, 1) if suggested_growth else None,
            'suggestedMargin': round(suggested_margin * 100, 1) if suggested_margin else None,
            'suggestedPE': round(suggested_pe, 1) if suggested_pe else None,
        }))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Intrinsic Value Endpoint ───

@app.route('/api/intrinsic/<ticker>')
def get_intrinsic_data(ticker):
    """Get financial data needed for intrinsic value calculation."""
    try:
        resolved = resolve_ticker(ticker.upper().strip())
        t = yf_ticker(resolved)
        info = t.info

        net_income = info.get('netIncomeToCommon', 0) or 0
        total_debt = info.get('totalDebt', 0) or 0
        total_cash = info.get('totalCash', 0) or 0
        shares = info.get('sharesOutstanding', 0) or 0
        price = safe(info, 'currentPrice', safe(info, 'regularMarketPrice', 0))
        earnings_growth = info.get('earningsGrowth')
        revenue_growth = info.get('revenueGrowth')

        # Try to get more precise data from income statement
        try:
            inc = t.income_stmt
            if inc is not None and not inc.empty:
                if 'Net Income' in inc.index:
                    ni_val = inc.loc['Net Income'].iloc[0]
                    if ni_val and not np.isnan(ni_val):
                        net_income = float(ni_val)
                elif 'Net Income Common Stockholders' in inc.index:
                    ni_val = inc.loc['Net Income Common Stockholders'].iloc[0]
                    if ni_val and not np.isnan(ni_val):
                        net_income = float(ni_val)
        except Exception:
            pass

        # Try balance sheet for more precise debt/cash
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                if 'Total Debt' in bs.index:
                    d = bs.loc['Total Debt'].iloc[0]
                    if d and not np.isnan(d):
                        total_debt = float(d)
                if 'Cash Cash Equivalents And Short Term Investments' in bs.index:
                    c = bs.loc['Cash Cash Equivalents And Short Term Investments'].iloc[0]
                    if c and not np.isnan(c):
                        total_cash = float(c)
        except Exception:
            pass

        # Suggest growth rates based on actual data
        suggested_growth_1_5 = min(abs(earnings_growth or revenue_growth or 0.15) * 100, 50)
        suggested_growth_6_10 = suggested_growth_1_5 / 2
        suggested_growth_11_20 = min(suggested_growth_6_10 / 2, 5)

        _is_ils = info.get('currency') == 'ILA' or '.TA' in resolved
        _agorot = 100 if info.get('currency') == 'ILA' else 1

        return jsonify(sanitize_for_json({
            'ticker': ticker.upper(),
            'name': safe(info, 'longName', ticker),
            'netIncome': net_income,
            'totalDebt': total_debt,
            'totalCash': total_cash,
            'sharesOutstanding': shares,
            'currentPrice': price,
            'earningsGrowth': earnings_growth,
            'revenueGrowth': revenue_growth,
            'suggestedGrowth1_5': round(suggested_growth_1_5, 1),
            'suggestedGrowth6_10': round(suggested_growth_6_10, 1),
            'suggestedGrowth11_20': round(suggested_growth_11_20, 1),
            'isILS': _is_ils,
            'currencySymbol': '₪' if _is_ils else '$',
            'agorotDivisor': _agorot,
            'financialCurrency': info.get('financialCurrency', 'USD'),
        }))

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Podcast Endpoints ───

import subprocess
import sys
from pathlib import Path

PODCAST_DIR = Path(__file__).parent.parent / "stock-podcast" / "podcasts"
PODCAST_SCRIPT = Path(__file__).parent.parent / "stock-podcast" / "podcast.py"


@app.route('/api/podcast/<ticker>', methods=['POST'])
def create_podcast(ticker):
    """Generate a podcast for a ticker."""
    try:
        ticker = ticker.upper().strip()
        resolved = resolve_ticker(ticker)

        # Run podcast.py as subprocess
        env = {**dict(__import__('os').environ)}
        result = subprocess.run(
            [sys.executable, str(PODCAST_SCRIPT), "--ticker", resolved],
            capture_output=True, text=True, timeout=120,
            cwd=str(PODCAST_SCRIPT.parent), env=env,
        )

        today = datetime.now().strftime("%Y-%m-%d")

        # Find generated files
        audio_file = PODCAST_DIR / f"{resolved}_{today}.mp3"
        script_file = PODCAST_DIR / f"{resolved}_{today}_script.txt"

        if not audio_file.exists() and not script_file.exists():
            return jsonify({'error': f'Podcast generation failed: {result.stderr[-300:] if result.stderr else "unknown error"}'}), 500

        # Get company name from yfinance
        try:
            company = yf_ticker(resolved).info.get('longName', ticker)
        except:
            company = ticker

        script_text = script_file.read_text(encoding='utf-8') if script_file.exists() else ''

        return jsonify({
            'ticker': ticker,
            'company': company,
            'date': today,
            'script': script_text,
            'has_audio': audio_file.exists(),
        })

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Podcast generation timed out (2 min)'}), 504
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/podcast/<ticker>/audio')
def get_podcast_audio(ticker):
    """Serve the podcast MP3 file."""
    try:
        ticker = resolve_ticker(ticker.upper().strip())
        today = datetime.now().strftime("%Y-%m-%d")
        audio_file = PODCAST_DIR / f"{ticker}_{today}.mp3"

        if not audio_file.exists():
            # Try yesterday
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            audio_file = PODCAST_DIR / f"{ticker}_{yesterday}.mp3"

        if not audio_file.exists():
            return jsonify({'error': 'No podcast audio found'}), 404

        return send_file(str(audio_file), mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/podcast/<ticker>/status')
def podcast_status(ticker):
    """Check if a podcast exists for today."""
    try:
        ticker_resolved = resolve_ticker(ticker.upper().strip())
        today = datetime.now().strftime("%Y-%m-%d")

        audio_file = PODCAST_DIR / f"{ticker_resolved}_{today}.mp3"
        script_file = PODCAST_DIR / f"{ticker_resolved}_{today}_script.txt"

        exists = audio_file.exists() or script_file.exists()

        company = ticker
        try:
            company = yf_ticker(ticker_resolved).info.get('longName', ticker)
        except:
            pass

        script_text = ''
        if script_file.exists():
            script_text = script_file.read_text(encoding='utf-8')

        return jsonify({
            'exists': exists,
            'has_audio': audio_file.exists(),
            'company': company,
            'date': today,
            'script': script_text,
        })
    except Exception as e:
        return jsonify({'exists': False, 'error': str(e)})


# Background crumb refresher — retries with exponential backoff if crumb is missing
def _crumb_refresh_loop():
    import time as _t
    _t.sleep(10)  # wait for startup
    backoff = 300  # start at 5 min
    max_backoff = 3600  # cap at 1 hour
    max_retries = 20  # stop after 20 consecutive failures
    failures = 0
    while True:
        try:
            if not _yahoo_crumb:
                print("[crumb-refresher] Attempting to get Yahoo crumb...")
                _init_yahoo_session(force_refresh=True)
                if _yahoo_crumb:
                    print(f"[crumb-refresher] Success! Crumb: {_yahoo_crumb[:8]}...")
                    backoff = 300  # reset backoff on success
                    failures = 0
                else:
                    failures += 1
                    if failures >= max_retries:
                        print(f"[crumb-refresher] Giving up after {max_retries} failures")
                        break
                    print(f"[crumb-refresher] Failed ({failures}/{max_retries}), will retry in {backoff}s")
                    backoff = min(backoff * 2, max_backoff)
            else:
                backoff = 300  # already have crumb, reset backoff
                failures = 0
        except Exception as e:
            failures += 1
            if failures >= max_retries:
                print(f"[crumb-refresher] Giving up after {max_retries} errors")
                break
            print(f"[crumb-refresher] Error: {e}")
            backoff = min(backoff * 2, max_backoff)
        _t.sleep(backoff)

import threading
_crumb_thread = threading.Thread(target=_crumb_refresh_loop, daemon=True)
_crumb_thread.start()

# Also try to init crumb at startup
try:
    _init_yahoo_session()
except Exception:
    pass

# ─── Daily Briefing API ───

@app.route('/api/briefing/generate', methods=['POST'])
def generate_briefing():
    """Generate a new daily briefing for given tickers."""
    try:
        data = request.get_json() or {}
        tickers = data.get('tickers', [])
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400

        from daily_briefing import collect_all, generate_text_report, generate_podcast_script, text_to_speech, TICKER_MAP, HEBREW_NAMES
        import daily_briefing
        daily_briefing.PORTFOLIO = tickers

        all_data = collect_all()
        report = generate_text_report(all_data)
        script = generate_podcast_script(all_data)

        today = datetime.now().strftime('%Y-%m-%d')
        briefings_dir = os.path.join(os.path.dirname(__file__), 'briefings')
        os.makedirs(briefings_dir, exist_ok=True)

        # Save files
        report_path = os.path.join(briefings_dir, f'briefing_{today}.txt')
        script_path = os.path.join(briefings_dir, f'briefing_{today}_script.txt')
        audio_path = os.path.join(briefings_dir, f'briefing_{today}.mp3')
        data_path = os.path.join(briefings_dir, f'briefing_{today}_data.json')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)

        text_to_speech(script, audio_path)

        return jsonify({
            'date': today,
            'report': report,
            'script': script,
            'hasAudio': os.path.exists(audio_path),
            'stocks': len([s for s in all_data['stocks'] if 'error' not in s]),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/briefing/latest')
def get_latest_briefing():
    """Get the latest briefing if it exists."""
    briefings_dir = os.path.join(os.path.dirname(__file__), 'briefings')
    today = datetime.now().strftime('%Y-%m-%d')
    report_path = os.path.join(briefings_dir, f'briefing_{today}.txt')
    script_path = os.path.join(briefings_dir, f'briefing_{today}_script.txt')
    audio_path = os.path.join(briefings_dir, f'briefing_{today}.mp3')

    if not os.path.exists(report_path):
        return jsonify({'exists': False})

    with open(report_path, 'r', encoding='utf-8') as f:
        report = f.read()

    script = ''
    if os.path.exists(script_path):
        with open(script_path, 'r', encoding='utf-8') as f:
            script = f.read()

    return jsonify({
        'exists': True,
        'date': today,
        'report': report,
        'script': script,
        'hasAudio': os.path.exists(audio_path),
    })


@app.route('/api/briefing/audio')
def serve_briefing_audio():
    """Serve the latest briefing audio file."""
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    audio_path = os.path.join(os.path.dirname(__file__), 'briefings', f'briefing_{date}.mp3')
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/mpeg')
    return jsonify({'error': 'No audio found'}), 404


# ─── Stock News Endpoint ───

@app.route('/api/stock-news/<ticker>')
def get_stock_news(ticker):
    """Fetch recent news for a stock using Tavily + Yahoo RSS.
    Returns a structured JSON summary.
    """
    ticker = ticker.upper().strip()

    try:
        news_items = []

        # Source 1: Yahoo Finance RSS (free, no key)
        try:
            import feedparser
            feed = feedparser.parse(f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US')
            for entry in feed.entries[:8]:
                news_items.append({
                    'title': entry.get('title', ''),
                    'source': 'Yahoo Finance',
                    'url': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', '')[:300],
                })
        except Exception as e:
            print(f"Yahoo RSS failed for {ticker}: {e}")

        # Source 2: Google News RSS (free, no key)
        try:
            import feedparser as fp2
            feed2 = fp2.parse(f'https://news.google.com/rss/search?q={ticker}+stock&hl=en&gl=US&ceid=US:en')
            for entry in feed2.entries[:6]:
                url = entry.get('link', '')
                # Deduplicate
                if not any(n['url'] == url for n in news_items):
                    source = entry.get('source', {})
                    source_name = source.get('title', 'Google News') if isinstance(source, dict) else 'Google News'
                    news_items.append({
                        'title': entry.get('title', ''),
                        'source': source_name,
                        'url': url,
                        'published': entry.get('published', ''),
                        'summary': '',
                    })
        except Exception as e:
            print(f"Google News RSS failed for {ticker}: {e}")

        # Source 3: Tavily deep search (API key required)
        tavily_key = os.environ.get('TAVILY_API_KEY', '')
        if tavily_key and not tavily_key.startswith('your_'):
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=tavily_key)
                response = client.search(
                    query=f"{ticker} stock news today",
                    search_depth="advanced",
                    max_results=6,
                )
                seen_urls = {n['url'] for n in news_items}
                for r in response.get('results', []):
                    url = r.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        domain = url.split('://')[1].split('/')[0].replace('www.', '') if '://' in url else 'Tavily'
                        news_items.append({
                            'title': r.get('title', ''),
                            'source': domain,
                            'url': url,
                            'published': '',
                            'summary': (r.get('content') or '')[:300],
                        })
            except Exception as e:
                print(f"Tavily failed for {ticker}: {e}")

        return jsonify({
            'ticker': ticker,
            'count': len(news_items),
            'news': news_items[:15],
            'fetched_at': datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5050))
    print("\n" + "="*50)
    print(f"  Stock Portfolio Tracker")
    print(f"  Open: http://localhost:{port}")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', debug=True, port=port)

