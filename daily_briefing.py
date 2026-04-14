#!/usr/bin/env python3
"""
Daily Portfolio Briefing — Text + Podcast
Generates a morning analyst briefing covering:
1. Macro overview (Fear & Greed, VIX, S&P, USD/ILS)
2. Per-stock analysis (price, news, fundamentals)
3. Written report + Hebrew audio podcast
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import yfinance as yf

# Load .env file if it exists
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ.setdefault(_key.strip(), _val.strip())

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

# ─── Configuration ───
PORTFOLIO = [
    'META', 'MSFT', 'SOFI', 'ADBE', 'AMZN',
    'PANW', 'ONON', 'IREN', 'MELI',
    'LEUMI', 'BEZEQ',  # Israeli stocks
]

OUTPUT_DIR = Path(__file__).parent / 'briefings'
OUTPUT_DIR.mkdir(exist_ok=True)

TICKER_MAP = {
    'LEUMI': 'LUMI.TA', 'POALIM': 'POLI.TA', 'HAPOALIM': 'POLI.TA',
    'BEZEQ': 'BEZQ.TA', 'AZRIELI': 'AZRG.TA', 'TEVA': 'TEVA.TA',
    'ADOBE': 'ADBE', 'GOOGLE': 'GOOGL', 'AMAZON': 'AMZN',
}

HEBREW_NAMES = {
    'META': 'מטא', 'MSFT': 'מייקרוסופט', 'SOFI': 'סופי',
    'ADBE': 'אדובי', 'AMZN': 'אמזון', 'PANW': 'פאלו אלטו',
    'ONON': 'און', 'IREN': 'אירן', 'MELI': 'מרקדו ליברה',
    'LEUMI': 'בנק לאומי', 'BEZEQ': 'בזק',
}

def _wilder_rsi(prices, period=14):
    """Compute RSI using Wilder's smoothing (the standard method used by TradingView, etc.)."""
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Seed with SMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Wilder's smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)


def resolve_ticker(t):
    return TICKER_MAP.get(t.upper(), t.upper())

def safe_get(d, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None

def _np_isnan(v):
    try:
        return np.isnan(v)
    except Exception:
        return False

# ─── Data Collection ───

def collect_macro():
    """Collect macro indicators."""
    print("📊 Collecting macro data...")
    macro = {}

    # Fear & Greed
    try:
        r = requests.get('https://production.dataviz.cnn.io/index/fearandgreed/graphdata',
                         headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        fg = r.json().get('fear_and_greed', {})
        macro['fear_greed'] = {
            'score': round(fg.get('score', 0)),
            'rating': fg.get('rating', ''),
            'prev_close': round(fg.get('previous_close', 0)),
            'week_ago': round(fg.get('previous_1_week', 0)),
            'month_ago': round(fg.get('previous_1_month', 0)),
        }
    except:
        macro['fear_greed'] = {'score': None}

    # VIX
    try:
        vix = yf.Ticker('^VIX')
        vi = vix.info
        macro['vix'] = round(safe_get(vi, 'regularMarketPrice', 'currentPrice') or 0, 1)
    except:
        macro['vix'] = None

    # S&P 500
    try:
        sp = yf.Ticker('^GSPC')
        si = sp.info
        sp_price = safe_get(si, 'regularMarketPrice', 'currentPrice')
        sp_prev = safe_get(si, 'previousClose', 'regularMarketPreviousClose')
        sp_change = round((sp_price - sp_prev) / sp_prev * 100, 2) if sp_price and sp_prev else 0

        # RSI (Wilder's smoothing)
        hist = sp.history(period='6mo', interval='1d')
        sp_rsi = None
        if not hist.empty and len(hist) > 30:
            sp_rsi = _wilder_rsi(hist['Close'].values, 14)

        macro['sp500'] = {
            'price': round(sp_price, 2) if sp_price else None,
            'change_pct': sp_change,
            'rsi': sp_rsi,
        }
    except:
        macro['sp500'] = {'price': None}

    # USD/ILS
    try:
        ils = yf.Ticker('ILS=X')
        ii = ils.info
        macro['usd_ils'] = round(safe_get(ii, 'regularMarketPrice', 'currentPrice') or 0, 4)
    except:
        macro['usd_ils'] = None

    # TA-35
    try:
        ta = yf.Ticker('TA35.TA')
        ti = ta.info
        ta_price = round(safe_get(ti, 'regularMarketPrice', 'currentPrice') or 0, 2)
        ta_prev = safe_get(ti, 'previousClose') or ta_price
        ta_chg = round((ta_price - ta_prev) / ta_prev * 100, 2) if ta_prev else 0
        macro['ta35'] = {'price': ta_price, 'change_pct': ta_chg}
    except:
        macro['ta35'] = None

    # 10Y Treasury Yield (^TNX) — key driver of tech/growth valuations
    try:
        tnx = yf.Ticker('^TNX')
        ti = tnx.info
        yield_now = safe_get(ti, 'regularMarketPrice', 'currentPrice')
        yield_prev = safe_get(ti, 'previousClose')
        if yield_now:
            change_bps = round((yield_now - yield_prev) * 100, 1) if yield_prev else 0
            macro['treasury_10y'] = {
                'yield': round(yield_now, 3),
                'change_bps': change_bps,
            }
    except:
        macro['treasury_10y'] = None

    # Key sector ETF moves — detect sector-wide pressure vs stock-specific moves
    try:
        sector_etfs = {'XLK': 'טק', 'XLF': 'פיננסים', 'XLV': 'בריאות', 'XLE': 'אנרגיה'}
        sector_moves = {}
        for etf, label in sector_etfs.items():
            try:
                et = yf.Ticker(etf)
                ei = et.info
                ep = safe_get(ei, 'regularMarketPrice', 'currentPrice')
                ec = safe_get(ei, 'previousClose')
                if ep and ec:
                    sector_moves[label] = round((ep - ec) / ec * 100, 2)
            except Exception:
                pass
        if sector_moves:
            macro['sector_etfs'] = sector_moves
    except:
        pass

    # Top macro/geopolitical news stories moving markets today (via Tavily)
    macro['macro_news'] = _collect_macro_news()
    if macro['macro_news']:
        print(f"  📰 Macro news: {len(macro['macro_news'])} stories")

    return macro


def collect_stock(ticker):
    """Collect data for a single stock."""
    resolved = resolve_ticker(ticker)
    t = yf.Ticker(resolved)
    info = t.info

    if not info:
        return {'ticker': ticker, 'error': 'No data'}

    is_ils = info.get('currency') == 'ILA'
    div = 100 if is_ils else 1
    cur = '₪' if is_ils else '$'

    price = safe_get(info, 'currentPrice', 'regularMarketPrice') or 0
    prev = safe_get(info, 'previousClose') or price
    change_pct = round((price - prev) / prev * 100, 2) if prev else 0

    # Fundamentals
    pe = safe_get(info, 'trailingPE')
    fwd_pe = safe_get(info, 'forwardPE')
    eps = safe_get(info, 'trailingEps')
    fwd_eps = safe_get(info, 'forwardEps')
    rev_growth = safe_get(info, 'revenueGrowth')
    earn_growth = safe_get(info, 'earningsGrowth')
    profit_margin = safe_get(info, 'profitMargins')
    market_cap = safe_get(info, 'marketCap')

    # Target price
    target_mean = safe_get(info, 'targetMeanPrice')
    target_low = safe_get(info, 'targetLowPrice')
    target_high = safe_get(info, 'targetHighPrice')
    recommendation = safe_get(info, 'recommendationKey')

    # Sector
    sector = info.get('sector') or info.get('sectorKey') or ''

    # RSI (Wilder's smoothing) + MA200 + 1Y performance
    hist_long = t.history(period='2y', interval='1d')
    rsi = None
    ma200 = None
    change_1y = None
    if not hist_long.empty and len(hist_long) > 30:
        rsi = _wilder_rsi(hist_long['Close'].values, 14)
    if not hist_long.empty and len(hist_long) >= 200:
        ma200_val = hist_long['Close'].rolling(200).mean().iloc[-1]
        if not np.isnan(ma200_val):
            ma200 = round(float(ma200_val), 2)
    if not hist_long.empty and len(hist_long) >= 252:
        price_1y_ago = float(hist_long['Close'].iloc[-252])
        if price_1y_ago > 0:
            change_1y = round((price - price_1y_ago) / price_1y_ago * 100, 1)

    # 52-week range
    w52_high = safe_get(info, 'fiftyTwoWeekHigh')
    w52_low = safe_get(info, 'fiftyTwoWeekLow')

    # Distance from 52w high
    pct_from_high = round((price - w52_high) / w52_high * 100, 1) if w52_high and price else None

    # Earnings calendar
    earnings_date = None
    earnings_est_eps = None
    earnings_est_revenue = None
    try:
        cal = t.calendar
        if cal is not None and isinstance(cal, dict):
            dates = cal.get('Earnings Date', [])
            if dates:
                d0 = dates[0]
                # yfinance may return datetime.date or pd.Timestamp
                earnings_date = d0.strftime('%Y-%m-%d') if hasattr(d0, 'strftime') else str(d0)[:10]
            earnings_est_eps = cal.get('Earnings Average')
            earnings_est_revenue = cal.get('Revenue Average')
    except Exception:
        pass

    # Recent analyst upgrades/downgrades (last 14 days)
    import datetime as _dt
    recent_analyst_actions = []
    try:
        upgrades = t.upgrades_downgrades
        if upgrades is not None and not upgrades.empty:
            upgrades = upgrades.sort_index(ascending=False)
            cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=14)
            for idx, row in upgrades.iterrows():
                # index may be tz-aware or tz-naive Timestamp
                try:
                    idx_dt = idx if idx.tzinfo else idx.tz_localize('UTC')
                    if idx_dt < cutoff:
                        break
                except Exception:
                    pass
                target = row.get('currentPriceTarget')
                recent_analyst_actions.append({
                    'date': str(idx)[:10],
                    'firm': str(row.get('Firm', '')),
                    'from_grade': str(row.get('FromGrade', '')),
                    'to_grade': str(row.get('ToGrade', '')),
                    'action': str(row.get('Action', '')),
                    'price_target': round(float(target), 0) if target and not _np_isnan(target) else None,
                })
                if len(recent_analyst_actions) >= 5:
                    break
    except Exception:
        pass

    # Short interest — squeeze potential signal
    short_pct = None
    short_ratio = None
    try:
        sp_float = safe_get(info, 'shortPercentOfFloat')
        if sp_float and sp_float > 0:
            short_pct = round(sp_float * 100, 1)  # e.g. 0.08 → 8.0%
        sr = safe_get(info, 'shortRatio')
        if sr:
            short_ratio = round(float(sr), 1)  # days to cover
    except Exception:
        pass

    # Insider transactions — last 90 days, significant moves only
    # yfinance columns: Shares, Value, Text, Insider, Position, Transaction, Start Date, Ownership
    insider_activity = []
    try:
        insiders = t.insider_transactions
        if insiders is not None and not insiders.empty:
            import datetime as _dt2
            cutoff = _dt2.datetime.now() - _dt2.timedelta(days=90)
            for _, row in insiders.iterrows():
                try:
                    tx_date = row['Start Date']
                    if hasattr(tx_date, 'to_pydatetime'):
                        tx_date = tx_date.to_pydatetime()
                    if tx_date.replace(tzinfo=None) < cutoff:
                        continue
                    value_tx = float(row.get('Value') or 0)
                    if abs(value_tx) < 100_000:  # skip tiny transactions
                        continue
                    tx_type = str(row.get('Transaction') or '')
                    # Derivative conversions are exercise, not open-market buys — skip
                    text_lower = str(row.get('Text') or '').lower()
                    if 'conversion' in text_lower or 'exercise' in text_lower or 'derivative' in text_lower:
                        continue
                    is_sale = any(w in tx_type.lower() for w in ['sale', 'sell', 'sold'])
                    is_buy = any(w in tx_type.lower() for w in ['purchase', 'buy', 'bought'])
                    if not is_sale and not is_buy:
                        continue
                    insider_activity.append({
                        'date': str(tx_date)[:10],
                        'name': str(row.get('Insider') or ''),
                        'relation': str(row.get('Position') or ''),
                        'type': 'מכירה' if is_sale else 'קנייה',
                        'value': int(abs(value_tx)),
                        'shares': int(abs(float(row.get('Shares') or 0))),
                    })
                    if len(insider_activity) >= 4:
                        break
                except Exception:
                    continue
    except Exception:
        pass

    # News from Yahoo RSS
    news = []
    try:
        import feedparser
        feed = feedparser.parse(f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={resolved}')
        for entry in feed.entries[:5]:
            news.append({
                'title': entry.get('title', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
            })
    except:
        pass

    return {
        'ticker': ticker,
        'resolved': resolved,
        'name': safe_get(info, 'longName', 'shortName') or ticker,
        'hebrew': HEBREW_NAMES.get(ticker, ticker),
        'price': round(price / div, 2),
        'change_pct': change_pct,
        'currency': cur,
        'pe': round(pe, 1) if pe else None,
        'forward_pe': round(fwd_pe, 1) if fwd_pe else None,
        'eps': round(eps, 2) if eps else None,
        'forward_eps': round(fwd_eps, 2) if fwd_eps else None,
        'revenue_growth': round(rev_growth * 100, 1) if rev_growth else None,
        'earnings_growth': round(earn_growth * 100, 1) if earn_growth else None,
        'profit_margin': round(profit_margin * 100, 1) if profit_margin else None,
        'market_cap': market_cap,
        'rsi': rsi,
        'ma200': round(ma200 / div, 2) if ma200 else None,
        'w52_high': round(w52_high / div, 2) if w52_high else None,
        'w52_low': round(w52_low / div, 2) if w52_low else None,
        'pct_from_high': pct_from_high,
        'target_mean': round(target_mean / div, 2) if target_mean else None,
        'target_low': round(target_low / div, 2) if target_low else None,
        'target_high': round(target_high / div, 2) if target_high else None,
        'recommendation': recommendation,
        'news': news,
        'prev_price': round(prev / div, 2),
        'earnings_date': earnings_date,
        'earnings_est_eps': round(earnings_est_eps, 2) if earnings_est_eps else None,
        'earnings_est_revenue': earnings_est_revenue,
        'recent_analyst_actions': recent_analyst_actions,
        'sector': sector,
        'change_1y': change_1y,
        'short_pct': short_pct,
        'short_ratio': short_ratio,
        'insider_activity': insider_activity,
    }


def _is_generic_url(url):
    """Filter out generic stock data pages that don't contain real news."""
    skip_domains = [
        'tradingview.com', 'stockanalysis.com', 'finance.yahoo.com/quote',
        'google.com/finance', 'marketwatch.com/investing/stock/',
        'wsj.com/market-data', 'macrotrends.net',
    ]
    url_lower = url.lower()
    return any(d in url_lower for d in skip_domains)


def _run_with_timeout(fn, timeout=15, default=None):
    """Run fn() with a hard timeout. Returns default on timeout/error.
    Does NOT wait for the hung thread — avoids blocking the caller.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        result = fut.result(timeout=timeout)
        ex.shutdown(wait=False)
        return result
    except FuturesTimeout:
        ex.shutdown(wait=False, cancel_futures=True)
        return default
    except Exception:
        ex.shutdown(wait=False)
        return default


def _tavily_extract_urls(tavily_client, urls):
    """Use Tavily extract to get full article content from URLs."""
    try:
        result = _run_with_timeout(
            lambda: tavily_client.extract(urls=urls[:5]),
            timeout=10, default={}
        )
        if not result:
            return {}
        extracted = {}
        for r in result.get('results', []):
            url = r.get('url', '')
            text = r.get('raw_content', '') or r.get('text', '')
            if url and text and len(text) > 100:
                extracted[url] = text[:3000]
        return extracted
    except Exception as e:
        print(f"    ⚠️  Tavily extract failed: {e}")
        return {}


# ── Curated expert X/Twitter accounts — high signal-to-noise, institutional-grade ──
# Sources: SentimenTrader "most useful FinTwit" ranking + The Bear Cave 100 list +
#          Acquired/All-In research community + semiconductor/tech specialist accounts.
_EXPERT_ACCOUNTS = [
    # Macro & rates — the people who move the narrative
    'KobeissiLetter',   # Adam Kobeissi — global macro, rates, equities; 700k followers, CNBC regular
    'elerianm',         # Mohamed El-Erian — Fed policy, central banking, global macro; ex-PIMCO CEO
    'NorthmanTrader',   # Sven Henrich — technical analysis, S&P structure, MarketWatch contributor
    'cullenroche',      # Cullen Roche — monetary mechanics, debunks myths; Orcam Financial founder
    'IanShepherdson',   # Ian Shepherdson — Pantheon Macro chief economist; Fed forecasting
    # Fundamental equity — deep valuation and business analysis
    'AswathDamodaran',  # "Dean of Valuation" — NYU professor; DCF models on any stock
    'charliebilello',   # Charlie Bilello — pure data + historical market stats; best quant visual feed
    'morganhousel',     # Morgan Housel — Psychology of Money author; behavioral finance
    'GavinSBaker',      # Gavin Baker — Atreides CIO; tech sector, semiconductors; ex-Fidelity OTC
    # Tech & semiconductors — specialist-level analysis
    'dylan522p',        # Dylan Patel — SemiAnalysis; semiconductor supply chain, NVDA/TSMC/Intel
    'benthompson',      # Ben Thompson — Stratechery; tech business strategy, platform economics
    # Options flow & market structure
    'unusual_whales',   # Options flow, dark pools, institutional positioning; real-time alerts
    'OptionsHawk',      # Joe Kunkle — real-time options order flow, sector sweeps
    'sentimentrader',   # SentimenTrader — sentiment indicators, historical analogues; ranked #1 quality
    # Commentary & behavioral
    'fundstrat',        # Tom Lee / Fundstrat — macro strategy, earnings season interpretation
    'ReformedBroker',   # Josh Brown — CNBC; market analysis + sharp cultural commentary
    'ritholtz',         # Barry Ritholtz — financial market history, behavioral economics
    'michaelbatnick',   # Michael Batnick — Ritholtz; investor mistakes, market data
    'awealthofcs',      # Ben Carlson — evidence-based investing, behavioral finance
]

# Spam/noise patterns — tweets matching these are penalized
_SPAM_PATTERNS = [
    '🚀🚀🚀', '100x', 'to the moon', 'buy now', 'free signal',
    'dm me', 'join my', 'guaranteed', 'get rich', '🔥🔥',
]


def _score_stocktwits(msg):
    """Quality score for a StockTwits message. Higher = better."""
    import re
    body = msg.get('body', '')
    score = 0
    # Length bonus — substance over noise
    if len(body) > 120:
        score += 3
    elif len(body) > 60:
        score += 1
    # Data-driven: contains numbers, percentages, dollar amounts
    if re.search(r'\d+\.?\d*%|\$\d+|\d+B|\d+M', body):
        score += 2
    # Has explicit sentiment tag — user has conviction
    if msg.get('entities', {}).get('sentiment', {}).get('basic'):
        score += 1
    # Spam/noise penalty
    body_lower = body.lower()
    for pat in _SPAM_PATTERNS:
        if pat.lower() in body_lower:
            score -= 4
    return score


def _collect_reddit(ticker, company_name=''):
    """Search Reddit for ticker via the public JSON API (no auth needed).
    Returns posts sorted by engagement score (upvotes × subreddit quality weight).
    """
    # (subreddit, quality_weight) — SecurityAnalysis has the deepest analysis
    subreddits = [
        ('SecurityAnalysis', 4),
        ('ValueInvesting', 3),
        ('stocks', 2),
        ('investing', 1),
    ]
    posts = []
    seen_ids = set()
    headers = {'User-Agent': 'PortfolioBriefingBot/1.0 (personal finance tool)'}

    def _fetch_sub(sub_weight):
        sub, weight = sub_weight
        results = []
        try:
            url = f'https://www.reddit.com/r/{sub}/search.json'
            params = {'q': ticker, 'sort': 'top', 't': 'week', 'limit': 10, 'restrict_sr': 1}
            resp = requests.get(url, params=params, headers=headers, timeout=6)
            if resp.status_code != 200:
                return results
            children = resp.json().get('data', {}).get('children', [])
            for child in children:
                d = child.get('data', {})
                post_id = d.get('id', '')
                title = d.get('title', '')
                selftext = d.get('selftext', '') or ''
                ups = d.get('ups', 0)
                combined = (title + ' ' + selftext).upper()
                if (ticker.upper() not in combined
                        and (not company_name or company_name.split()[0].upper() not in combined)):
                    continue
                if len(title) < 15:
                    continue
                if ups < 5 and sub not in ('SecurityAnalysis', 'ValueInvesting'):
                    continue
                results.append((post_id, {
                    'title': title,
                    'text': selftext[:600] if len(selftext) > 50 else '',
                    'ups': ups,
                    'score': ups * weight,
                    'subreddit': sub,
                    'permalink': f"reddit.com{d.get('permalink', '')}",
                }))
        except Exception:
            pass
        return results

    # Sequential subreddit requests — requests.get handles timeout at socket level
    for sw in subreddits:
        for post_id, post in _fetch_sub(sw):
            if post_id not in seen_ids:
                seen_ids.add(post_id)
                posts.append(post)

    # Deduplicate by title — same story can appear in multiple subreddits
    posts.sort(key=lambda x: x['score'], reverse=True)
    seen_titles = set()
    deduped = []
    for p in posts:
        title_key = p['title'].lower()[:60]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            deduped.append(p)
    return deduped[:5]


def _collect_macro_news():
    """Fetch today's top market-moving macro/geopolitical news — free, no API key.

    Sources (in priority order):
      1. Google News RSS — dynamic search on 3 topics
      2. yfinance news for macro proxies (S&P, Oil, Treasury)
    Returns top 6 stories scored by financial keyword density.
    Each story: {'title', 'snippet', 'theme'}
    """
    _MACRO_KWS = [
        'market', 'stock', 'oil', 'crude', 'opec', 'fed', 'rate', 'rates',
        'inflation', 'tariff', 'tariffs', 'sanction', 'war', 'iran', 'israel',
        'china', 'trade', 'recession', 'gdp', 'jobs', 'cpi', 'earnings',
        'rally', 'selloff', 'sell-off', 'crash', 'surge', 'plunge', 'spike',
        'treasury', 'yield', 'dollar', 'gold', 'geopolit', 'conflict',
    ]

    candidates = []
    seen = set()

    def _strip_html(text):
        """Remove HTML tags and decode basic entities."""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    def _add(title, snippet, theme, boost=0):
        title = _strip_html(title)
        snippet = _strip_html(snippet) if snippet else ''
        key = title.lower()[:55]
        if key in seen or not title:
            return
        seen.add(key)
        text_low = (title + ' ' + snippet).lower()
        score = sum(1 for kw in _MACRO_KWS if kw in text_low) + boost
        if score >= 2:
            candidates.append({'title': title, 'snippet': snippet, 'theme': theme, 'score': score})

    # ── Source 1: Google News RSS (free, no key) ──
    if feedparser:
        gnews_queries = [
            ("geopolitical", "geopolitical+oil+war+sanctions+markets"),
            ("market_moves", "stock+market+selloff+rally+today+macro"),
            ("fed_rates",    "Federal+Reserve+inflation+CPI+interest+rates"),
        ]
        for theme, q in gnews_queries:
            try:
                feed = _run_with_timeout(
                    lambda u=f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en": feedparser.parse(u),
                    timeout=8, default=None
                )
                if not feed:
                    continue
                for entry in (feed.entries or [])[:6]:
                    title = (entry.get('title') or '').strip()
                    summary = (entry.get('summary') or '').strip()
                    snippet = _best_news_sentence(summary) if summary else ''
                    _add(title, snippet, theme, boost=1)
            except Exception:
                pass

    # ── Source 2: yfinance news for macro proxies ──
    macro_proxies = [
        ('^GSPC', 'market_moves'),   # S&P 500
        ('CL=F',  'geopolitical'),   # Crude oil
        ('^TNX',  'fed_rates'),      # 10Y Treasury
        ('GC=F',  'geopolitical'),   # Gold
    ]
    for proxy_ticker, theme in macro_proxies:
        try:
            news_items = _run_with_timeout(
                lambda t=proxy_ticker: yf.Ticker(t).news,
                timeout=6, default=[]
            ) or []
            for item in (news_items or [])[:4]:
                title = (item.get('title') or '').strip()
                content = (item.get('summary') or item.get('content') or '').strip()
                snippet = _best_news_sentence(content) if content else ''
                _add(title, snippet, theme)
        except Exception:
            pass

    candidates.sort(key=lambda x: -x['score'])
    return candidates[:6]


def _collect_expert_tweets(ticker):
    """Search for recent expert X/Twitter posts about ticker via Tavily web search.
    Uses Tavily instead of Nitter — reliable, thread-safe, no zombie threads.
    Falls back to empty list gracefully if Tavily unavailable or no results found.
    """
    try:
        from tavily import TavilyClient
        api_key = os.environ.get('TAVILY_API_KEY', '')
        if not api_key:
            return []
        tavily = TavilyClient(api_key=api_key)

        # Build a targeted query: ticker + known expert handles on X
        # Use include_domains to restrict results to x.com only
        expert_handles = [
            'KobeissiLetter', 'unusual_whales', 'AswathDamodaran', 'charliebilello',
            'fundstrat', 'NorthmanTrader', 'GavinSBaker', 'dylan522p',
        ]
        handles_str = ' '.join(expert_handles[:2])
        query = f'{ticker} {handles_str} site:x.com'

        result = _run_with_timeout(
            lambda: tavily.search(query=query, search_depth="basic", max_results=8),
            timeout=8, default={}
        ) or {}

        posts = []
        seen_urls = set()
        for r in (result.get('results') or []):
            url = r.get('url', '')
            if url in seen_urls:
                continue
            # Only x.com/twitter.com results
            if 'x.com/' not in url and 'twitter.com/' not in url:
                continue
            seen_urls.add(url)
            content = (r.get('content') or r.get('snippet') or '').strip()
            if not content or len(content) < 40:
                continue

            # Extract author from URL (prefer direct tweet URLs)
            author = ''
            for domain in ('x.com/', 'twitter.com/'):
                if domain in url:
                    parts = url.split(domain)[-1].split('/')
                    if parts and parts[0] not in ('i', 'search', 'hashtag', ''):
                        author = parts[0]
                    break

            # If not from a known expert URL, scan content for expert mention
            is_expert = author.lower() in [h.lower() for h in expert_handles]
            if not is_expert:
                for handle in expert_handles:
                    if handle.lower() in content.lower():
                        author = handle
                        is_expert = True
                        break

            if author and is_expert:
                posts.append({
                    'author': f'@{author}',
                    'text': content[:280],
                    'url': url,
                })

        return posts[:3]
    except Exception:
        return []


def _collect_viral_tweets(ticker, company_name=''):
    """Fetch recent high-engagement tweets about a ticker using Twitter API v2.

    Requires TWITTER_BEARER_TOKEN in env.
    Returns up to 5 tweets sorted by engagement (likes + retweets + replies),
    filtered to a minimum buzz threshold to cut noise.
    """
    bearer = os.environ.get('TWITTER_BEARER_TOKEN', '')
    if not bearer:
        return []
    try:
        import tweepy

        client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=False)

        # Build query: ticker + optional company name, exclude retweets and replies,
        # English or Hebrew only, minimum 10 likes to cut pure noise
        q_terms = f'({ticker}'
        if company_name:
            q_terms += f' OR "{company_name}"'
        q_terms += ') -is:retweet -is:reply lang:en'

        result = _run_with_timeout(
            lambda: client.search_recent_tweets(
                query=q_terms,
                max_results=50,              # fetch 50, then rank by engagement
                tweet_fields=['public_metrics', 'created_at', 'author_id', 'text'],
                expansions=['author_id'],
                user_fields=['username', 'name', 'verified', 'public_metrics'],
            ),
            timeout=10, default=None
        )
        if not result or not result.data:
            return []

        # Build author lookup: author_id → username
        author_map = {}
        if result.includes and result.includes.get('users'):
            for u in result.includes['users']:
                author_map[u.id] = {
                    'username': u.username,
                    'followers': (u.public_metrics or {}).get('followers_count', 0),
                }

        # Score each tweet by engagement + author credibility
        scored = []
        for t in result.data:
            m = t.public_metrics or {}
            likes     = m.get('like_count', 0)
            retweets  = m.get('retweet_count', 0)
            replies   = m.get('reply_count', 0)
            quotes    = m.get('quote_count', 0)
            engagement = likes + (retweets * 2) + replies + quotes  # retweets weighted more

            if engagement < 15:   # skip low-signal tweets entirely
                continue

            author_info = author_map.get(t.author_id, {})
            username = author_info.get('username', 'unknown')
            followers = author_info.get('followers', 0)

            # Boost score for known expert accounts
            expert_boost = 50 if username.lower() in [h.lower() for h in _EXPERT_ACCOUNTS] else 0
            # Boost for accounts with significant following
            follower_boost = min(followers // 10000, 30)  # up to +30 for large accounts

            final_score = engagement + expert_boost + follower_boost

            scored.append({
                'author': f'@{username}',
                'text': t.text,
                'likes': likes,
                'retweets': retweets,
                'engagement': engagement,
                'score': final_score,
                'is_expert': expert_boost > 0,
            })

        # Sort by final score descending
        scored.sort(key=lambda x: x['score'], reverse=True)

        # Return top 5 (mix of expert + viral)
        return scored[:5]

    except Exception as e:
        print(f"  ⚠️  Twitter viral search failed ({type(e).__name__}): {e}")
        return []


def collect_social(ticker, company_name=''):
    """Collect social intelligence — four quality layers:
    1. Reddit (engagement-ranked, expert subreddits, no auth needed)
    2. StockTwits (quality-scored, spam-filtered sentiment)
    3. Viral X tweets via Twitter API v2 (requires TWITTER_BEARER_TOKEN)
    4. Expert X posts via Tavily search (fallback when no Bearer Token)
    """
    social = {'reddit': [], 'stocktwits': [], 'twitter': [], 'viral_tweets': [], 'sentiment': None}

    # ── Layer 1: Reddit ──
    social['reddit'] = _collect_reddit(ticker, company_name)

    # ── Layer 2: StockTwits — sentiment signal ──
    try:
        r = requests.get(
            f'https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json',
            headers={'User-Agent': 'Mozilla/5.0'}, timeout=(3, 5)
        )
        if r.status_code == 200:
            messages = r.json().get('messages', [])
            bullish = sum(1 for m in messages
                         if m.get('entities', {}).get('sentiment', {}).get('basic') == 'Bullish')
            bearish = sum(1 for m in messages
                         if m.get('entities', {}).get('sentiment', {}).get('basic') == 'Bearish')
            total = bullish + bearish
            if total > 0:
                social['sentiment'] = {
                    'bullish': bullish,
                    'bearish': bearish,
                    'bullish_pct': round(bullish / total * 100),
                    'bearish_pct': round(bearish / total * 100),
                    'total_messages': len(messages),
                }
            scored = sorted(
                [m for m in messages if len(m.get('body', '')) > 40],
                key=_score_stocktwits, reverse=True,
            )
            for m in scored[:6]:
                body = m.get('body', '').strip()
                social['stocktwits'].append({
                    'text': body,
                    'sentiment': m.get('entities', {}).get('sentiment', {}).get('basic', ''),
                    'user': m.get('user', {}).get('username', ''),
                })
    except Exception:
        pass

    # ── Layer 3: Viral tweets via Twitter API v2 (if Bearer Token configured) ──
    if os.environ.get('TWITTER_BEARER_TOKEN'):
        social['viral_tweets'] = _collect_viral_tweets(ticker, company_name)
        # Expert accounts that appeared in viral results go into 'twitter' too
        social['twitter'] = [t for t in social['viral_tweets'] if t.get('is_expert')]

    # ── Layer 4: Tavily expert X posts (fallback when no viral tweets) ──
    if not social['twitter']:
        social['twitter'] = _collect_expert_tweets(ticker)

    return social


def _feedparser_with_timeout(url, timeout=8, **kwargs):
    """feedparser via requests — thread-safe, proper socket-level timeout, no zombie threads."""
    empty = type('FP', (), {'entries': []})()
    if not feedparser:
        return empty
    try:
        resp = requests.get(url, timeout=(3, timeout),
                            headers={'User-Agent': 'Mozilla/5.0 (feedparser)'})
        if resp.status_code == 200:
            return feedparser.parse(resp.content)
    except Exception:
        pass
    return empty


def collect_news_deep(ticker, resolved_ticker, company_name, hebrew_name):
    """Collect news from 3 sources with actual article content."""
    articles = []
    seen_urls = set()
    rss_urls_to_extract = []  # URLs from RSS that need full content

    # Source 1: Yahoo Finance RSS
    if feedparser:
        try:
            feed = _feedparser_with_timeout(
                f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={resolved_ticker}')
            for entry in feed.entries[:8]:
                url = entry.get('link', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append({
                        'title': entry.get('title', ''),
                        'source': 'Yahoo Finance',
                        'url': url,
                        'published': entry.get('published', ''),
                        'content': entry.get('summary', ''),
                    })
                    rss_urls_to_extract.append(url)
        except Exception as e:
            print(f"    ⚠️  Yahoo RSS failed for {ticker}: {e}")

    # Source 2: Google News RSS
    if feedparser:
        try:
            gn_url = f'https://news.google.com/rss/search?q={ticker}+stock&hl=en&gl=US&ceid=US:en'
            feed = _feedparser_with_timeout(gn_url)
            for entry in feed.entries[:6]:
                url = entry.get('link', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    source = ''
                    if hasattr(entry, 'source'):
                        source = entry.source.get('title', 'Google News')
                    articles.append({
                        'title': entry.get('title', ''),
                        'source': source or 'Google News',
                        'url': url,
                        'published': entry.get('published', ''),
                        'content': entry.get('summary', ''),
                    })
                    rss_urls_to_extract.append(url)
        except Exception as e:
            print(f"    ⚠️  Google News RSS failed for {ticker}: {e}")

    # Tavily extract + search — sequential with hard timeouts (no nested pools)
    tavily_key = os.environ.get('TAVILY_API_KEY', '')
    if tavily_key and TavilyClient:
        tavily = TavilyClient(api_key=tavily_key)
        query = f"{company_name} ({ticker}) stock news latest developments"

        # Extract full content from RSS URLs
        if rss_urls_to_extract:
            extracted = _tavily_extract_urls(tavily, rss_urls_to_extract[:5]) or {}
            for a in articles:
                if a.get('url') in extracted:
                    a['content'] = extracted[a['url']]

        # Search for additional news
        results = _run_with_timeout(
            lambda: tavily.search(query=query, search_depth="advanced",
                                  max_results=5, include_raw_content=True),
            timeout=10, default={}
        ) or {}
        for r in results.get('results', []):
            url = r.get('url', '')
            if url and url not in seen_urls and not _is_generic_url(url):
                seen_urls.add(url)
                content = r.get('raw_content', '') or r.get('content', '')
                articles.append({
                    'title': r.get('title', ''),
                    'source': 'Tavily',
                    'url': url,
                    'published': '',
                    'content': content[:3000],
                })

    # For Israeli stocks, also search in Hebrew
    is_israeli = resolved_ticker.endswith('.TA')
    if is_israeli and tavily_key and TavilyClient:
        try:
            tavily = TavilyClient(api_key=tavily_key)
            query = f"{hebrew_name} מניה חדשות"
            results = _run_with_timeout(
                lambda: tavily.search(query=query, search_depth="advanced", max_results=3,
                                      include_raw_content=True),
                timeout=10, default={}
            ) or {}
            for r in results.get('results', []):
                url = r.get('url', '')
                if url and url not in seen_urls and not _is_generic_url(url):
                    seen_urls.add(url)
                    content = r.get('raw_content', '') or r.get('content', '')
                    articles.append({
                        'title': r.get('title', ''),
                        'source': 'Tavily (HE)',
                        'url': url,
                        'published': '',
                        'content': content[:3000],
                    })
        except Exception as e:
            print(f"    ⚠️  Tavily HE failed for {ticker}: {e}")

    return articles


OPENAI_MODEL = "gpt-4.1-mini"
DEEPSEEK_MODEL = "deepseek-chat"
GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _openai_chat(api_key, messages, max_tokens=1000, system=None):
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    if system:
        messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=OPENAI_MODEL, max_tokens=max_tokens, messages=messages,
    )
    return response.choices[0].message.content


def _deepseek_chat(api_key, messages, max_tokens=1000, system=None):
    """Call DeepSeek API (OpenAI-compatible)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    if system:
        messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL, max_tokens=max_tokens, messages=messages,
    )
    return response.choices[0].message.content


def _gemini_chat(api_key, messages, max_tokens=1000, system=None):
    """Call Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system if system else None,
    )

    gemini_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({"role": role, "parts": [msg["content"]]})

    response = model.generate_content(
        gemini_contents,
        generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
    )
    return response.text


def _groq_chat(api_key, messages, max_tokens=1000, system=None):
    """Call Groq API (fallback)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    if system:
        messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=GROQ_MODEL, max_tokens=max_tokens, messages=messages,
    )
    return response.choices[0].message.content


def _llm_chat(messages, max_tokens=1000, system=None):
    """Call LLM — tries OpenAI first, then DeepSeek, then Gemini, then Groq."""
    openai_key = os.environ.get('OPENAI_API_KEY', '')
    deepseek_key = os.environ.get('DEEPSEEK_API_KEY', '')
    gemini_key = os.environ.get('GEMINI_API_KEY', '')
    groq_key = os.environ.get('GROQ_API_KEY', '')

    # 1. OpenAI (primary)
    if openai_key:
        try:
            return _openai_chat(openai_key, messages, max_tokens, system)
        except Exception as e:
            print(f"    ⚠️  OpenAI failed: {e}")

    # 2. DeepSeek (fallback)
    if deepseek_key:
        try:
            print(f"    🔄 Trying DeepSeek...")
            return _deepseek_chat(deepseek_key, messages, max_tokens, system)
        except Exception as e:
            print(f"    ⚠️  DeepSeek failed: {e}")

    # 3. Gemini (fallback)
    if gemini_key:
        try:
            print(f"    🔄 Trying Gemini...")
            return _gemini_chat(gemini_key, messages, max_tokens, system)
        except Exception as e:
            print(f"    ⚠️  Gemini failed: {e}")

    # 4. Groq (last resort)
    if groq_key:
        print(f"    🔄 Trying Groq...")
        return _groq_chat(groq_key, messages, max_tokens, system)

    raise RuntimeError("No LLM API key configured (set OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY)")


_CAUSAL_WORDS = [
    'because', 'after', 'amid', 'due to', 'following', 'driven by',
    'cited', 'despite', 'as investors', 'as the company', 'as shares',
    'reported', 'announced', 'beat', 'missed', 'raised', 'cut', 'warned',
    'upgraded', 'downgraded', 'acquired', 'launched', 'partnered',
]

def _best_news_sentence(content):
    """Return the most informative sentence — prefer causal/event sentences over generic openers."""
    # Reject entire content if it looks like a scraped HTML page
    content_low = content.lower()
    if any(marker in content_low for marker in ['skip to navigation', 'oops, something went wrong',
                                                  'enable javascript', '<html', 'document.write']):
        return ''
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if len(s.strip()) > 50]
    if not sentences:
        return ''
    # Score each sentence: causal words = +2, longer = better up to 250 chars, boilerplate = -5
    boilerplate = ['click here', 'subscribe', 'sign up', 'cookie', 'privacy policy', 'read more',
                   'skip to navigation', 'skip to content', 'oops, something went wrong',
                   'javascript is required', 'enable javascript', 'please enable',
                   'terms of service', 'terms of use', 'all rights reserved', 'advertisement']
    html_garbage = ['<html', '<head', '<body', '<div', '<script', 'document.write', 'window.location']
    scored = []
    for sent in sentences[:15]:  # scan first 15 sentences
        low = sent.lower()
        # Skip boilerplate and HTML fragments
        if any(b in low for b in boilerplate):
            continue
        if any(h in low for h in html_garbage):
            continue
        # Skip sentences that are mostly bracket-wrapped content (scraped nav links)
        bracket_ratio = (sent.count('[') + sent.count(']')) / max(len(sent), 1)
        if bracket_ratio > 0.05:
            continue
        score = sum(2 for w in _CAUSAL_WORDS if w in low)
        score += min(len(sent), 250) / 50  # length bonus, capped
        scored.append((score, sent))
    if not scored:
        # Last resort: find first sentence without HTML markers
        clean_fallback = [s for s in sentences if not any(b in s.lower() for b in boilerplate + html_garbage)
                          and (s.count('[') + s.count(']')) / max(len(s), 1) <= 0.05]
        return clean_fallback[0][:280] if clean_fallback else ''
    scored.sort(key=lambda x: -x[0])
    return scored[0][1][:280]


def _extract_news_context(articles):
    """Deterministic news extraction — headlines + best causal sentence. No LLM needed.
    Returns same shape as summarize_stock_news for compatibility with podcast builder.
    """
    articles_with_content = [a for a in articles if a.get('content') and len(a['content']) > 50]
    best = articles_with_content or articles
    if not best:
        return {'has_news': False, 'article_count': 0, 'key_headlines': [], 'summary': ''}

    lines = []
    for a in best[:4]:
        title = a.get('title', '')
        source = a.get('source', '')
        content = a.get('content', '')
        snippet = _best_news_sentence(content) if content else ''
        entry = f"• {title} ({source})"
        if snippet:
            entry += f"\n  {snippet}"
        lines.append(entry)

    return {
        'has_news': True,
        'article_count': len(articles),
        'key_headlines': [a['title'] for a in best[:3]],
        'summary': '\n'.join(lines),
    }


def summarize_stock_news(ticker, hebrew_name, stock_data, articles, social=None):
    """DEPRECATED — kept for reference only. Use _extract_news_context instead."""
    articles_with_content = [a for a in articles if a.get('content') and len(a['content']) > 50]
    social = social or {}
    has_social = bool(social.get('reddit') or social.get('stocktwits') or social.get('twitter'))

    if not articles_with_content and not has_social:
        return {
            'summary': '',
            'has_news': False,
            'article_count': len(articles),
            'key_headlines': [a['title'] for a in articles[:3]],
        }

    try:
        change_pct = stock_data.get('change_pct', 0)

        # Earnings + analyst context
        events_text = ""
        if stock_data.get('earnings_date'):
            try:
                from datetime import date as _date
                ed = datetime.strptime(stock_data['earnings_date'], '%Y-%m-%d').date()
                days_away = (ed - _date.today()).days
                if -7 <= days_away <= 60:
                    timing = (f"בעוד {days_away} יום" if days_away > 0
                              else f"לפני {abs(days_away)} יום" if days_away < 0
                              else "היום")
                    eps_str = f", קונצנזוס EPS: ${stock_data['earnings_est_eps']}" if stock_data.get('earnings_est_eps') else ""
                    events_text += f"\nדוח רווחים קרוב: {ed.strftime('%d/%m/%Y')} ({timing}){eps_str}\n"
            except Exception:
                pass

        analyst_actions = stock_data.get('recent_analyst_actions', [])
        if analyst_actions:
            events_text += "\nשינויי דירוג אנליסטים (14 יום אחרונים):\n"
            for a in analyst_actions[:4]:
                grade_change = f"{a['from_grade']} → {a['to_grade']}" if a['from_grade'] and a['to_grade'] else a['to_grade']
                pt_str = f" | יעד מחיר: ${a['price_target']:,.0f}" if a.get('price_target') else ""
                events_text += f"- {a['date']} | {a['firm']}: {a['action']} ({grade_change}){pt_str}\n"

        articles_text = ""
        for i, a in enumerate(articles_with_content[:5], 1):
            content = a['content'][:2000]
            articles_text += f"\n--- כתבה {i}: {a['title']} ({a['source']}) ---\n{content}\n"

        # Build social section for prompt
        social_text = ""

        # Reddit — highest quality, show top posts with upvote count
        reddit_posts = social.get('reddit', [])
        if reddit_posts:
            social_text += "\n--- דיון ברדיט (ממוין לפי engagement) ---\n"
            for p in reddit_posts[:3]:
                sub_label = f"r/{p['subreddit']}"
                ups_label = f"↑{p['ups']:,}" if p['ups'] > 0 else ""
                body_preview = f" — {p['text'][:200]}" if p.get('text') else ""
                social_text += f"[{sub_label} {ups_label}] {p['title']}{body_preview}\n"

        # StockTwits sentiment signal
        sent = social.get('sentiment')
        if sent:
            social_text += f"\n--- סנטימנט StockTwits ({sent['total_messages']} הודעות) ---\n"
            social_text += f"{sent['bullish_pct']}% Bullish | {sent['bearish_pct']}% Bearish\n"
        st_posts = social.get('stocktwits', [])
        if st_posts:
            social_text += "הודעות איכותיות מ-StockTwits:\n"
            for p in st_posts[:4]:
                tag = f" [{p['sentiment']}]" if p.get('sentiment') else ""
                social_text += f"- {p['text'][:200]}{tag}\n"

        # Expert Twitter — labeled as credible voices
        tw_posts = social.get('twitter', [])
        if tw_posts:
            social_text += "\n--- ציוצים ממומחים ידועים (X/Twitter) ---\n"
            for p in tw_posts[:4]:
                social_text += f"@{p['author']}: {p['text'][:220]}\n"

        prompt = f"""אתה מגיש פודקאסט כלכלי יומי בעברית. הסגנון שלך כמו מגיש פודקאסט ישראלי טוב — מקצועי, ברור, מדבר בגובה העיניים. לא קריין חדשות רשמי, אבל גם לא חבר'ה מהשכונה.

המניה: {hebrew_name} ({ticker})
מחיר: {stock_data.get('currency', '$')}{stock_data.get('price', '?')} | שינוי: {change_pct:+.1f}%
{events_text}{articles_text}{social_text}

כתוב 200-350 מילים. זה פודקאסט — טקסט שמיועד להקראה בקול.

הנחיות:
- ספר את הסיפור של המניה היום כנרטיב אחד זורם. אסור "הכתבה הראשונה", "כתבה נוספת", "לפי כתבה ש..."
- שזור את כל המידע מהכתבות ביחד — מה קרה, למה, ומה המשמעות
- אם יש דוח רווחים קרוב — ציין את התאריך ואת קונצנזוס האנליסטים. זה הקשר חשוב מאוד
- אם היה שינוי דירוג של אנליסט לאחרונה — ציין בשם: "גולדמן סאקס שדרג ל-Buy" וכדומה
- אם יש דיון ברדיט עם upvotes גבוהים — זה signal שמשהו תופס תשומת לב. שזור זאת בסיפור
- אם מומחה ידוע (ציוצי המומחים) אמר משהו ספציפי — ציין שמו ומה אמר, זה מידע בעל ערך
- אם הסנטימנט ב-StockTwits חריג (80%+ Bullish או Bearish) — ציין זאת כחלק מהסיפור
- ציין פרטים ספציפיים: שמות אנשים, מספרים, סכומים, אירועים
- תן הקשר — למה זה חשוב, מה זה אומר על החברה
- סיים עם תובנה אחת — מה הדבר הכי חשוב לזכור מהיום
- טון: מקצועי אבל נגיש. לא סלנג ("אחי", "מגניב", "ענק"), לא שפה יבשה ("יש לציין", "ראוי להדגיש")
- תרגם הכל לעברית
- אסור: כוכביות, מספור, markdown, כותרות באנגלית"""

        summary = _llm_chat([{"role": "user", "content": prompt}], max_tokens=1500)
        return {
            'summary': summary,
            'has_news': True,
            'article_count': len(articles),
            'key_headlines': [a['title'] for a in articles_with_content[:3]],
        }

    except Exception as e:
        print(f"    ⚠️  News summarization failed for {ticker}: {e}")
        return {
            'summary': '',
            'has_news': False,
            'article_count': len(articles),
            'key_headlines': [a['title'] for a in articles[:3]],
        }


def collect_all():
    """Collect all data."""
    print("=" * 60)
    print(f"🎙️  Daily Portfolio Briefing — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    macro = collect_macro()

    stocks = []
    for ticker in PORTFOLIO:
        print(f"  📈 {ticker}...", end=" ", flush=True)
        try:
            data = collect_stock(ticker)
            stocks.append(data)
            print(f"✅ {data['currency']}{data['price']} ({data['change_pct']:+.1f}%)")
        except Exception as e:
            print(f"❌ {e}")
            stocks.append({'ticker': ticker, 'error': str(e)})

    # Phase 2: News + social collection in parallel
    print("\n📰 Collecting news + social (parallel)...")
    valid_stocks = [s for s in stocks if 'error' not in s]

    def _collect_news_and_social(s):
        ticker = s['ticker']
        articles = collect_news_deep(ticker, s['resolved'], s.get('name', ''), s.get('hebrew', ''))
        social = collect_social(ticker, s.get('name', ''))
        return ticker, articles, social

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results_map = {}
    collect_pool = ThreadPoolExecutor(max_workers=5)
    try:
        futs = {collect_pool.submit(_collect_news_and_social, s): s['ticker'] for s in valid_stocks}
        try:
            for fut in as_completed(futs, timeout=120):
                try:
                    ticker, articles, social = fut.result()
                    results_map[ticker] = (articles, social)
                    content_count = len([a for a in articles if a.get('content') and len(a['content']) > 50])
                    reddit_count = len(social.get('reddit', []))
                    st_count = len(social.get('stocktwits', []))
                    sent = social.get('sentiment')
                    sent_str = f" {sent['bullish_pct']}%🟢" if sent else ""
                    print(f"  ✅ {ticker}: {len(articles)}art/{content_count}full | Reddit:{reddit_count} ST:{st_count}{sent_str}")
                except Exception as e:
                    print(f"  ⚠️  {futs[fut]}: {e}")
        except Exception:
            print("  ⚠️  Collection timeout — proceeding with partial results")
    finally:
        collect_pool.shutdown(wait=False, cancel_futures=True)

    # Attach data + deterministic news context (no LLM — podcast LLM handles synthesis)
    for s in valid_stocks:
        ticker = s['ticker']
        articles, social = results_map.get(ticker, ([], {}))
        s['news_deep'] = articles
        s['social'] = social
        s['news_summary'] = _extract_news_context(articles)
        if s['news_summary']['has_news']:
            print(f"  📋 {ticker}: {s['news_summary']['article_count']} articles extracted")

    return {'macro': macro, 'stocks': stocks, 'date': datetime.now().isoformat()}


# ─── Report Generation ───

def format_large(v):
    if not v:
        return '—'
    if abs(v) >= 1e12:
        return f'{v/1e12:.1f}T'
    if abs(v) >= 1e9:
        return f'{v/1e9:.1f}B'
    if abs(v) >= 1e6:
        return f'{v/1e6:.0f}M'
    return f'{v:,.0f}'


def generate_text_report(data, positions=None):
    """Generate Hebrew text report. positions = {ticker: {shares: N}} or {ticker: N}"""
    m = data['macro']
    stocks = data['stocks']
    date = datetime.fromisoformat(data['date']).strftime('%d/%m/%Y')

    # Normalize positions: accept both {ticker: N} and {ticker: {shares: N}}
    if positions:
        positions = {
            t: (v if isinstance(v, dict) else {'shares': v})
            for t, v in positions.items()
        }

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"📊 סיכום יומי לתיק ההשקעות — {date}")
    lines.append(f"{'='*60}")
    lines.append("")

    # ─── Portfolio P&L summary ───
    if positions:
        valid_pos = {t: v for t, v in positions.items() if v.get('shares', 0) > 0}
        if valid_pos:
            total_value = 0
            total_pnl = 0
            pnl_rows = []
            for s in [s for s in stocks if 'error' not in s and s['ticker'] in valid_pos]:
                pos = valid_pos[s['ticker']]
                shares = pos['shares']
                entry_price = pos.get('entry_price')
                prev = s.get('prev_price', s['price'])
                pnl = shares * (s['price'] - prev)
                value = shares * s['price']
                total_pnl += pnl
                total_value += value
                s['_dollar_impact'] = pnl
                s['_shares'] = shares
                # Cost basis for unrealized P&L
                if entry_price:
                    s['_entry_price'] = entry_price
                    s['_cost_basis'] = entry_price * shares
                    s['_unrealized_pnl'] = (s['price'] - entry_price) * shares
                    s['_unrealized_pct'] = (s['price'] - entry_price) / entry_price * 100
                pnl_rows.append((s, shares, pnl, value))

            pnl_sign = '+' if total_pnl >= 0 else ''
            pnl_emoji = '🟢' if total_pnl >= 0 else '🔴'
            lines.append("💼 תמצית תיק ההשקעות היום")
            lines.append("─" * 40)
            lines.append(f"  💰 שווי כולל:   ${total_value:,.0f}")
            lines.append(f"  {pnl_emoji} שינוי יומי:  {pnl_sign}${total_pnl:,.0f}  ({pnl_sign}{(total_pnl/total_value*100) if total_value else 0:.2f}%)")
            # Total unrealized P&L (cost basis)
            unrealized_stocks = [s for s, *_ in pnl_rows if '_unrealized_pnl' in s]
            if unrealized_stocks:
                total_unrealized = sum(s['_unrealized_pnl'] for s in unrealized_stocks)
                total_cost = sum(s['_cost_basis'] for s in unrealized_stocks)
                unr_sign = '+' if total_unrealized >= 0 else ''
                unr_pct = (total_unrealized / total_cost * 100) if total_cost else 0
                unr_emoji = '🟢' if total_unrealized >= 0 else '🔴'
                lines.append(f"  {unr_emoji} רווח/הפסד כולל:  {unr_sign}${total_unrealized:,.0f}  ({unr_sign}{unr_pct:.1f}% מעלות בסיס)")
            lines.append("")

            # Top movers by dollar impact
            pnl_rows_sorted = sorted(pnl_rows, key=lambda x: x[2])
            if any(r[2] < 0 for r in pnl_rows_sorted):
                lines.append("  📉 הכי השפיעו לרעה:")
                for s, shares, pnl, value in [r for r in pnl_rows_sorted if r[2] < 0][:3]:
                    lines.append(f"    {s['hebrew']} ({s['ticker']}): {pnl:+,.0f}$ ({s['change_pct']:+.1f}% × {shares:g} מניות)")
            if any(r[2] > 0 for r in pnl_rows_sorted):
                lines.append("  📈 הכי השפיעו לטובה:")
                for s, shares, pnl, value in sorted([r for r in pnl_rows_sorted if r[2] > 0], key=lambda x: -x[2])[:3]:
                    lines.append(f"    {s['hebrew']} ({s['ticker']}): {pnl:+,.0f}$ ({s['change_pct']:+.1f}% × {shares:g} מניות)")
            lines.append("")

    # ─── Macro ───
    lines.append("🌍 מאקרו — תמונת מצב השוק")
    lines.append("─" * 40)

    fg = m.get('fear_greed', {})
    if fg.get('score') is not None:
        score = fg['score']
        emoji = '🟢' if score > 60 else '🟡' if score > 40 else '🔴'
        lines.append(f"  {emoji} Fear & Greed: {score} ({fg.get('rating', '')})")
        lines.append(f"     לפני שבוע: {fg.get('week_ago', '—')} | לפני חודש: {fg.get('month_ago', '—')}")

    sp = m.get('sp500', {})
    if sp.get('price'):
        sp_emoji = '🟢' if sp['change_pct'] > 0 else '🔴'
        lines.append(f"  {sp_emoji} S&P 500: {sp['price']:,.2f} ({sp['change_pct']:+.2f}%)")
        if sp.get('rsi'):
            rsi_status = 'Oversold 🔴' if sp['rsi'] < 30 else 'Overbought 🟢' if sp['rsi'] > 70 else 'Neutral'
            lines.append(f"     RSI: {sp['rsi']} — {rsi_status}")

    if m.get('vix'):
        vix_emoji = '🔴' if m['vix'] > 25 else '🟡' if m['vix'] > 18 else '🟢'
        lines.append(f"  {vix_emoji} VIX: {m['vix']} {'(גבוה — פחד בשוק)' if m['vix'] > 25 else '(רגוע)' if m['vix'] < 18 else ''}")

    if m.get('usd_ils'):
        lines.append(f"  💱 USD/ILS: ₪{m['usd_ils']:.4f}")

    if m.get('ta35'):
        ta35 = m['ta35']
        lines.append(f"  🇮🇱 TA-35: {ta35['price']:,.2f} ({ta35['change_pct']:+.2f}%)")

    lines.append("")

    # ─── Summary signal ───
    fg_score = fg.get('score') or 50
    vix_val = m.get('vix') or 20

    if fg_score < 20 and vix_val > 30:
        lines.append("⚠️  אזהרה: פחד קיצוני בשוק — VIX גבוה + Fear & Greed נמוך מאוד")
        lines.append("   יכול להיות הזדמנות קנייה למשקיעים לטווח ארוך")
    elif fg_score > 75:
        lines.append("⚠️  זהירות: תאוות בצע בשוק — שקול לקחת רווחים")

    lines.append("")

    # ─── Stocks ───
    lines.append("📈 ניתוח מניות בתיק")
    lines.append("─" * 40)

    valid_stocks = [s for s in stocks if 'error' not in s]
    # Sort by dollar impact if positions provided, else by % change
    if positions and any('_dollar_impact' in s for s in valid_stocks):
        valid_stocks.sort(key=lambda s: s.get('_dollar_impact', 0))
    else:
        valid_stocks.sort(key=lambda s: s.get('change_pct', 0))

    # Winners & Losers
    winners = [s for s in valid_stocks if s.get('change_pct', 0) > 0]
    losers = [s for s in valid_stocks if s.get('change_pct', 0) <= 0]

    def fmt_stock_line(s):
        base = f"    {s['hebrew']} ({s['ticker']}): {s['currency']}{s['price']} ({s['change_pct']:+.1f}%)"
        if '_dollar_impact' in s and '_shares' in s:
            impact = s['_dollar_impact']
            base += f"  →  {'+' if impact >= 0 else ''}{impact:,.0f}$"
        return base

    if losers:
        lines.append("")
        lines.append("  🔴 ירידות:")
        for s in losers:
            lines.append(fmt_stock_line(s))

    if winners:
        lines.append("")
        lines.append("  🟢 עליות:")
        for s in reversed(winners):
            lines.append(fmt_stock_line(s))

    lines.append("")

    # ─── Per-stock detail ───
    lines.append("📋 ניתוח מפורט לכל מניה")
    lines.append("═" * 50)

    for s in valid_stocks:
        lines.append("")
        lines.append(f"  ▸ {s['hebrew']} ({s['ticker']}) — {s['currency']}{s['price']} ({s['change_pct']:+.1f}%)")

        details = []
        if s.get('pe'):
            details.append(f"P/E: {s['pe']}x")
        if s.get('forward_pe'):
            details.append(f"Forward P/E: {s['forward_pe']}x")
        if s.get('revenue_growth'):
            details.append(f"צמיחת הכנסות: {s['revenue_growth']:+.1f}%")
        if s.get('earnings_growth'):
            details.append(f"צמיחת רווח: {s['earnings_growth']:+.1f}%")
        if s.get('profit_margin'):
            details.append(f"מרווח רווח: {s['profit_margin']:.1f}%")
        if details:
            lines.append(f"    {' | '.join(details)}")

        # RSI + MA200 + 52W
        sub = []
        if s.get('rsi'):
            rsi_txt = 'Oversold 🔴' if s['rsi'] < 30 else 'Overbought 🟢' if s['rsi'] > 70 else ''
            sub.append(f"RSI(14): {s['rsi']}{' (' + rsi_txt + ')' if rsi_txt else ''}")
        if s.get('ma200'):
            above_below = 'מעל' if s['price'] > s['ma200'] else 'מתחת'
            pct_from_ma = round((s['price'] - s['ma200']) / s['ma200'] * 100, 1)
            sub.append(f"MA200: {s['currency']}{s['ma200']} ({above_below}, {pct_from_ma:+.1f}%)")
        if s.get('pct_from_high'):
            sub.append(f"מרחק מ-52W High: {s['pct_from_high']:+.1f}%")
        if sub:
            lines.append(f"    {' | '.join(sub)}")

        # Analyst target
        if s.get('target_mean'):
            upside = round((s['target_mean'] - s['price']) / s['price'] * 100, 1)
            lines.append(f"    🎯 יעד אנליסטים: {s['currency']}{s['target_mean']} ({upside:+.1f}% upside) | המלצה: {s.get('recommendation', '—')}")

        # Earnings calendar
        if s.get('earnings_date'):
            from datetime import date as _date
            try:
                ed = datetime.strptime(s['earnings_date'], '%Y-%m-%d').date()
                days_away = (ed - _date.today()).days
                if -7 <= days_away <= 60:
                    timing = (f"בעוד {days_away} יום" if days_away > 0
                              else f"לפני {abs(days_away)} יום" if days_away < 0
                              else "היום")
                    eps_str = f" | EPS קונצנזוס: ${s['earnings_est_eps']}" if s.get('earnings_est_eps') else ""
                    lines.append(f"    📅 דוח רווחים: {ed.strftime('%d/%m/%Y')} ({timing}){eps_str}")
            except Exception:
                pass

        # Recent analyst upgrades/downgrades
        analyst_actions = s.get('recent_analyst_actions', [])
        if analyst_actions:
            lines.append(f"    📊 שינויי דירוג אחרונים:")
            for a in analyst_actions[:3]:
                action_map = {'up': '⬆️ שדרוג', 'down': '⬇️ הורדה', 'init': '🆕 כיסוי חדש', 'reit': '➡️ אחזקה', 'main': '➡️ אחזקה'}
                action_label = action_map.get(a['action'].lower(), a['action'])
                grade_change = f"{a['from_grade']} → {a['to_grade']}" if a['from_grade'] and a['to_grade'] else a['to_grade']
                pt_str = f" | יעד: ${a['price_target']:,.0f}" if a.get('price_target') else ""
                lines.append(f"      {a['date']} | {a['firm']}: {action_label} ({grade_change}){pt_str}")

        # Social intelligence
        social = s.get('social', {})
        if not isinstance(social, dict):
            print(f"  ⚠️  DEBUG: social for {s.get('ticker')} is {type(social).__name__}: {repr(social)[:100]}")
            social = {}

        # Reddit — top posts by engagement
        reddit_posts = social.get('reddit', [])
        if reddit_posts:
            lines.append(f"    🟠 Reddit (top posts):")
            for p in reddit_posts[:2]:
                ups_str = f" ↑{p['ups']:,}" if p['ups'] > 0 else ""
                lines.append(f"      [{p['subreddit']}{ups_str}] {p['title'][:110]}")

        # StockTwits sentiment bar
        sent = social.get('sentiment')
        if sent:
            bull_bar = '█' * (sent['bullish_pct'] // 10) + '░' * (10 - sent['bullish_pct'] // 10)
            lines.append(f"    💬 סנטימנט ({sent['total_messages']} הודעות): 🟢{sent['bullish_pct']}% {bull_bar} 🔴{sent['bearish_pct']}%")

        # Expert tweets (high-credibility)
        tw_posts = social.get('twitter', [])
        if tw_posts:
            lines.append(f"    𝕏 מומחים:")
            for p in tw_posts[:2]:
                lines.append(f"      @{p['author']}: {p['text'][:110]}")

        # News summary
        news_summary = s.get('news_summary', {})
        if news_summary.get('has_news') and news_summary.get('summary'):
            lines.append(f"    📰 סיכום חדשות ({news_summary.get('article_count', 0)} כתבות):")
            for line in news_summary['summary'].split('\n'):
                if line.strip():
                    lines.append(f"    {line.strip()}")
        elif s.get('news'):
            lines.append(f"    📰 {len(s['news'])} כתבות חדשות פורסמו היום")

    lines.append("")
    lines.append("─" * 40)
    lines.append("⚠️  אין זו המלצת השקעה. נא להתייעץ עם יועץ מורשה.")
    lines.append(f"נוצר: {datetime.now().strftime('%H:%M %d/%m/%Y')}")

    return '\n'.join(lines)




def generate_notebooklm_sources(data, positions=None):
    """Generate narrative source for NotebookLM podcast + article URLs.

    Returns:
        dict with 'text' (str) and 'urls' (list of str)
    """
    m = data['macro']
    stocks = [s for s in data['stocks'] if 'error' not in s]
    date_str = datetime.fromisoformat(data['date']).strftime('%d/%m/%Y')
    article_urls = []

    lines = []
    lines.append(f"סקירה יומית של שוק ההון — {date_str}")
    lines.append("")

    # ─── Market Narrative — tone matches the actual data ───
    sp = m.get('sp500', {})
    sp_price = sp.get('price', 0)
    sp_chg = sp.get('change_pct', 0)
    vix = m.get('vix', 0)
    fg = m.get('fear_greed', {})
    fg_val = fg.get('value', 0)
    fg_label = fg.get('label', '')
    usd_ils_raw = m.get('usd_ils', 0)
    usd_ils = usd_ils_raw.get('rate', 0) if isinstance(usd_ils_raw, dict) else usd_ils_raw
    treasury = m.get('treasury_10y', {})

    # Write the market description in a tone proportional to what actually happened
    if abs(sp_chg) < 0.3:
        lines.append(f"יום רגוע יחסית בוול סטריט. ה-S&P 500 סגר כמעט ללא שינוי, {'עלייה' if sp_chg >= 0 else 'ירידה'} של {abs(sp_chg):.1f}% בלבד, על {sp_price:,.0f} נקודות. אין פה דרמה מיוחדת במדדים הראשיים.")
    elif sp_chg > 1.5:
        lines.append(f"יום ראלי חזק בשוק. ה-S&P 500 זינק ב-{sp_chg:.1f}% ל-{sp_price:,.0f} נקודות. המשקיעים באופוריה.")
    elif sp_chg > 0.3:
        lines.append(f"יום חיובי בשוק. ה-S&P 500 עלה ב-{sp_chg:.1f}% ל-{sp_price:,.0f} נקודות.")
    elif sp_chg < -1.5:
        lines.append(f"יום ירידות חזקות בשוק. ה-S&P 500 נפל ב-{abs(sp_chg):.1f}% ל-{sp_price:,.0f} נקודות. המשקיעים מודאגים.")
    else:
        lines.append(f"יום שלילי מתון בשוק. ה-S&P 500 ירד ב-{abs(sp_chg):.1f}% ל-{sp_price:,.0f} נקודות. ירידה לא דרמטית, אבל שווה לשים לב למגמה.")

    if vix > 30:
        lines.append(f"ה-VIX ברמה של {vix:.0f}. השוק מתנהג כמו נהג שרואה אורות אדומים בכל צומת — כולם דורכים על הברקס.")
    elif vix > 25:
        lines.append(f"ה-VIX ברמה גבוהה של {vix:.0f}. סוחרי האופציות קונים הגנות באגרסיביות — מישהו מריח בעיה.")
    elif vix > 18:
        lines.append(f"ה-VIX ברמה של {vix:.0f}. לא פאניקה, אבל גם לא שאננות. השוק על המשמר.")
    else:
        lines.append(f"ה-VIX ברמה נמוכה של {vix:.0f}. המשקיעים ישנים טוב בלילה — לפחות בינתיים.")

    if fg_val:
        if fg_val < 20:
            lines.append(f"מדד הפחד והחמדנות צנח ל-{fg_val:.0f}. וורן באפט אוהב לומר: תהיה חמדן כשאחרים פוחדים. אנחנו שם עכשיו.")
        elif fg_val < 35:
            lines.append(f"מדד הפחד והחמדנות ברמה נמוכה של {fg_val:.0f} — פחד. היסטורית, רגעים כאלה היו הזדמנויות קנייה.")
        elif fg_val > 80:
            lines.append(f"מדד הפחד והחמדנות ב-{fg_val:.0f} — חמדנות קיצונית. כשכולם בטוחים שהם גאונים, בדרך כלל מגיע התיקון.")
        elif fg_val > 65:
            lines.append(f"מדד הפחד והחמדנות ב-{fg_val:.0f} — חמדנות. האופטימיות גבוהה, אבל שווה לזכור שהשוק לא עולה לנצח.")

    if treasury.get('yield_pct'):
        y = treasury['yield_pct']
        lines.append(f"תשואת האג\"ח ל-10 שנים עומדת על {y:.2f}%. {'תשואה גבוהה שלוחצת על מניות צמיחה.' if y > 4.5 else ''}")

    if usd_ils:
        lines.append(f"שער הדולר-שקל: {usd_ils:.2f}.")

    # Macro news — filter out headlines that contradict the actual data
    macro_news = m.get('macro_news', [])
    if macro_news:
        crash_words = ['crash', 'plunge', 'collapse', 'meltdown', 'bloodbath', 'tank', 'crater']
        rally_words = ['soar', 'surge', 'skyrocket', 'boom', 'explode']
        filtered_news = []
        for n in macro_news[:6]:
            title = n.get('title', '')
            if not title:
                continue
            title_lower = title.lower()
            # Don't include "market crash" headlines on a flat day
            if abs(sp_chg) < 1.0 and any(w in title_lower for w in crash_words + rally_words):
                continue
            filtered_news.append(title)
        if filtered_news:
            lines.append("")
            lines.append("כותרות מהעולם:")
            for title in filtered_news[:4]:
                lines.append(f"- {title}")
    lines.append("")

    # ─── Find connections between stocks ───
    tickers_str = ', '.join(s.get('hebrew') or s.get('name', s.get('ticker', '')) for s in stocks)
    lines.append(f"המניות שנסקור היום: {tickers_str}.")
    lines.append("")

    # Check if all are approaching earnings
    earnings_stocks = []
    for s in stocks:
        ed = s.get('earnings_date', '')
        if ed:
            try:
                days = (datetime.strptime(str(ed)[:10], '%Y-%m-%d') - datetime.now()).days
                if 0 < days <= 30:
                    earnings_stocks.append((s.get('hebrew') or s.get('name', ''), days))
            except Exception:
                pass
    if len(earnings_stocks) >= 2:
        names = ' ו-'.join(e[0] for e in earnings_stocks)
        min_days = min(e[1] for e in earnings_stocks)
        if min_days <= 7:
            lines.append(f"עונת הדוחות מתחילה בעוד ימים ספורים. {names} עומדות לחשוף את הקלפים. השבועות הקרובים יגדירו את כיוון התיק לשארית השנה.")
        else:
            lines.append(f"{names} מדווחות על רווחים בקרוב. בעוד {min_days} יום מתחילה עונת האמת — הרגע שבו הסיפורים נגמרים והמספרים מדברים.")
        lines.append("")

    # ─── Per-Stock Rich Narrative ───
    for s in stocks:
        t = s.get('ticker', '')
        name = s.get('hebrew') or s.get('name', t)
        price = s.get('price', 0) or 0
        change = s.get('change_pct', 0) or 0
        pe = s.get('pe')
        fwd_pe = s.get('forward_pe')
        target = s.get('target_mean')
        rec = s.get('recommendation', '')
        pct_from_high = s.get('pct_from_high')
        earnings_date = s.get('earnings_date', '')
        rev_growth = s.get('revenue_growth')
        earn_growth = s.get('earnings_growth')
        profit_margin = s.get('profit_margin')
        sector = s.get('sector', '')

        lines.append(f"=== {name} ({t}) ===")
        lines.append("")

        # Opening narrative — what happened and context
        if abs(change) > 3:
            lines.append(f"{name} זזה בצורה חדה היום — {'עלייה' if change > 0 else 'ירידה'} של {abs(change):.1f}%. המניה נסחרת ב-{price:.0f} דולר.")
        elif abs(change) > 0.5:
            lines.append(f"{name} {'עלתה' if change > 0 else 'ירדה'} ב-{abs(change):.1f}% ל-{price:.0f} דולר.")
        else:
            lines.append(f"{name} כמעט לא זזה היום ונסחרת ב-{price:.0f} דולר.")

        # Where the stock stands — distance from high
        if pct_from_high and pct_from_high < -40:
            lines.append(f"המניה איבדה {abs(pct_from_high):.0f}% מהשיא. מחצית מהערך נמחקה. השאלה: האם זו חברה שבורה, או מניה במבצע?")
        elif pct_from_high and pct_from_high < -20:
            lines.append(f"המניה נמצאת {abs(pct_from_high):.0f}% מתחת לשיא. ירידה רצינית — אבל לפעמים דווקא משם מגיעות ההזדמנויות הגדולות.")
        elif pct_from_high and pct_from_high > -5:
            lines.append(f"המניה קרובה לשיא שלה — רק {abs(pct_from_high):.0f}% מתחת. מי שהחזיק, מחייך.")

        # Valuation narrative
        if pe and fwd_pe:
            if fwd_pe < pe * 0.7:
                lines.append(f"נתון שקופץ לעיניים: מכפיל הרווח הנוכחי הוא {pe:.0f}, אבל העתידי יורד ל-{fwd_pe:.0f}. השוק מתמחר קפיצה רצינית ברווחים — אם זה יקרה, המניה זולה. אם לא, היא מלכודת.")
            elif fwd_pe > pe * 1.2:
                lines.append(f"מכפיל הרווח עומד על {pe:.0f}, אבל העתידי עולה ל-{fwd_pe:.0f}. השוק מצפה לירידה ברווחים — סימן שצריך לשים לב.")
            else:
                lines.append(f"מכפיל הרווח עומד על {pe:.0f} (עתידי: {fwd_pe:.0f}).")

        # Growth
        if rev_growth and rev_growth > 20:
            lines.append(f"החברה צומחת מהר — הכנסות עלו ב-{rev_growth:.0f}%{'.' if not earn_growth else f', ורווחים ב-{earn_growth:.0f}%.'}")
        elif rev_growth:
            lines.append(f"צמיחת הכנסות: {rev_growth:.0f}%.")

        if profit_margin and profit_margin > 30:
            lines.append(f"מרווח רווח של {profit_margin:.0f}% — רווחיות גבוהה מאוד.")

        # Analyst opinion — this is key content
        if target and price:
            upside = ((target - price) / price * 100)
            if upside > 80:
                lines.append(f"האנליסטים חושבים שהמניה שווה {target:.0f} דולר — {upside:.0f}% מעל המחיר הנוכחי. או שכל וול סטריט טועה, או שזו אחת ההזדמנויות של השנה. המלצה: {rec}.")
            elif upside > 40:
                lines.append(f"פער חריג בין המחיר ליעד: האנליסטים רואים {target:.0f} דולר ({upside:.0f}% upside). כסף חכם או אופטימיות מופרזת? המלצה: {rec}.")
            elif upside > 20:
                lines.append(f"יעד אנליסטים: {target:.0f} דולר ({upside:.0f}% upside). פוטנציאל סולידי. המלצה: {rec}.")
            elif upside > 0:
                lines.append(f"יעד אנליסטים: {target:.0f} דולר ({upside:.0f}% upside). המלצה: {rec}.")

        # Recent analyst actions
        actions = s.get('recent_analyst_actions', [])
        if actions:
            for a in actions[:2]:
                firm = a.get('firm', '')
                to_grade = a.get('to_grade', '')
                a_target = a.get('target', '')
                if firm and to_grade:
                    target_str = f" עם יעד מחיר של {a_target} דולר" if a_target else ""
                    lines.append(f"לאחרונה, {firm} נתנו דירוג {to_grade}{target_str}.")

        # Earnings countdown with context
        if earnings_date:
            try:
                ed = datetime.strptime(str(earnings_date)[:10], '%Y-%m-%d')
                days_to = (ed - datetime.now()).days
                if 0 < days_to <= 30:
                    est_eps = s.get('earnings_est_eps')
                    if days_to <= 7:
                        lines.append(f"ספירה לאחור: דוח רווחים בעוד {days_to} ימים בלבד ({ed.strftime('%d/%m')}). {'EPS צפוי: ' + str(est_eps) + '.' if est_eps else ''} הרגע הזה יגדיר את הכיוון לחודשים הקרובים.")
                    elif days_to <= 21:
                        lines.append(f"דוח רווחים בעוד {days_to} יום ({ed.strftime('%d/%m')}). {'הקונצנזוס: EPS של ' + str(est_eps) + '.' if est_eps else ''} המספרים ידברו בקרוב — ואז נדע אם האופטימיות מוצדקת.")
                    else:
                        lines.append(f"דוח רווחים בעוד {days_to} יום ({ed.strftime('%d/%m')}). {'EPS צפוי: ' + str(est_eps) + '.' if est_eps else ''}")
            except Exception:
                pass

        # Short interest
        short_pct = s.get('short_pct')
        if short_pct and short_pct > 15:
            lines.append(f"{short_pct:.0f}% מהמניות בשורט. כל חמישית מניה מהולכת נגדה. מישהו מאוד בטוח שהמניה הזו תיפול — אבל אם הם טועים, ה-short squeeze יהיה אדיר.")
        elif short_pct and short_pct > 8:
            lines.append(f"שורט של {short_pct:.0f}%. יש קבוצה רצינית שמהמרת נגד המניה הזו. מה הם יודעים שאנחנו לא?")
        elif short_pct and short_pct > 5:
            lines.append(f"שורט של {short_pct:.0f}%. לחץ מוכרים בחסר על המניה.")

        # Insider activity
        insiders = s.get('insider_activity', [])
        if insiders:
            buys = [i for i in insiders if 'buy' in str(i.get('type', '')).lower() or 'purchase' in str(i.get('type', '')).lower()]
            sells = [i for i in insiders if 'sell' in str(i.get('type', '')).lower() or 'sale' in str(i.get('type', '')).lower()]
            if buys and not sells:
                lines.append(f"סימן חזק: אינסיידרים קנו {len(buys)} פעמים ב-3 חודשים האחרונים. כשהמנכ\"ל שם את הכסף שלו — זה אומר משהו שאף דוח אנליסט לא יכול.")
            elif sells and not buys:
                lines.append(f"אינסיידרים מוכרים: {len(sells)} מכירות ב-90 יום. לא בהכרח שלילי — לפעמים מוכרים כדי לממן בית חדש — אבל כשזה קורה לצד ירידות, שווה לשים לב.")

        # Social sentiment
        social = s.get('social', {})
        st_sentiment = social.get('stocktwits_sentiment', {})
        if st_sentiment.get('bullish_pct'):
            bp = st_sentiment['bullish_pct']
            bearp = st_sentiment.get('bearish_pct', 0)
            if bp > 70:
                lines.append(f"הסנטימנט ברשתות החברתיות מאוד שורי: {bp:.0f}% אופטימיים ב-StockTwits.")
            elif bearp > 50:
                lines.append(f"הסנטימנט ברשתות דובי: רק {bp:.0f}% אופטימיים ב-StockTwits.")
            else:
                lines.append(f"סנטימנט מעורב ב-StockTwits: {bp:.0f}% שוריים, {bearp:.0f}% דוביים.")

        reddit = social.get('reddit', [])
        if reddit:
            top = reddit[0]
            upvotes = top.get('upvotes', 0) or top.get('score', 0) or 0
            if upvotes > 100:
                lines.append(f"פוסט פופולרי ברדיט ({upvotes} upvotes): \"{top.get('title', '')}\".")

        # News — filter for relevance and add narrative
        news = s.get('news_deep', [])
        relevant_news = []
        for n in news:
            title = (n.get('title', '') or '').lower()
            # Filter: title should mention the ticker or company name
            name_lower = name.lower()
            ticker_lower = t.lower()
            company_name = (s.get('name', '') or '').lower()
            if ticker_lower in title or name_lower in title or company_name in title:
                relevant_news.append(n)

        if relevant_news:
            lines.append("")
            lines.append("חדשות עיקריות:")
            for n in relevant_news[:3]:
                title = n.get('title', '')
                content = n.get('content', '')
                if title:
                    lines.append(f"- {title}")
                    # Add first meaningful sentence of content if available
                    if content and len(content) > 50:
                        first_sentence = content.split('.')[0].strip()
                        if len(first_sentence) > 30 and len(first_sentence) < 200:
                            lines.append(f"  ({first_sentence}.)")

                url = n.get('url', '')
                if url and url.startswith('http') and 'google.com/rss' not in url:
                    article_urls.append(url)
        elif news:
            # Fallback: show top news even if not perfectly filtered
            lines.append("")
            lines.append("חדשות קשורות:")
            for n in news[:2]:
                title = n.get('title', '')
                if title:
                    lines.append(f"- {title}")
                url = n.get('url', '')
                if url and url.startswith('http') and 'google.com/rss' not in url:
                    article_urls.append(url)

        lines.append("")

    # ─── Discussion Points ───
    lines.append("נקודות מעניינות לדיון:")
    lines.append("")
    for s in stocks:
        t = s.get('ticker', '')
        name = s.get('hebrew') or s.get('name', t)
        change = s.get('change_pct', 0) or 0
        pct_from_high = s.get('pct_from_high')
        target = s.get('target_mean')
        price = s.get('price', 0) or 0
        short_pct = s.get('short_pct')

        if target and price:
            upside = ((target - price) / price * 100)
            if upside > 50:
                lines.append(f"- פער חריג בין המחיר ליעד של {name}: האנליסטים רואים {upside:.0f}% upside. האם השוק טועה, או שהאנליסטים לא מעדכנים את היעדים?")
        if pct_from_high and pct_from_high < -30:
            lines.append(f"- {name} ירדה {abs(pct_from_high):.0f}% מהשיא. זו הזדמנות קנייה עם מרווח ביטחון, או סימן שמשהו השתנה מהותית?")
        if short_pct and short_pct > 10:
            lines.append(f"- {short_pct:.0f}% שורט ב-{name}. מה יודעים המוכרים בחסר שהשוק לא?")
        if abs(change) > 4:
            lines.append(f"- {name} זזה {abs(change):.1f}% ביום אחד. מהלך חריג — האם זה מוצדק?")

    lines.append("")

    return {
        'text': '\n'.join(lines),
        'urls': article_urls[:9],
    }


def generate_podcast_script(data, user_name=None):
    """Generate Hebrew podcast script — tries LLM API first, falls back to basic."""
    has_llm = os.environ.get('DEEPSEEK_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GROQ_API_KEY', '')

    if has_llm:
        try:
            return _generate_podcast_with_llm(data, user_name=user_name)
        except Exception as e:
            print(f"  ⚠️  LLM API failed for briefing: {e}, using fallback")

    return _generate_podcast_fallback(data)


def _compute_analytical_signals(valid_stocks, macro):
    """Pre-compute analytical insights from raw data. Returns dict with market_signals and stock_signals."""
    market_signals = []
    stock_signals = {}  # ticker → list of strings

    sp500_change = macro.get('sp500', {}).get('change_pct', 0)

    # ── Signal 3: Fear & Greed momentum ──
    fg = macro.get('fear_greed', {})
    fg_now = fg.get('score')
    fg_week = fg.get('week_ago')
    fg_month = fg.get('month_ago')
    if fg_now is not None and fg_week is not None and abs(fg_now - fg_week) > 10:
        direction = 'ירד' if fg_now < fg_week else 'עלה'
        market_signals.append(f"סנטימנט השוק {direction} חדות — מ-{fg_week} לפני שבוע ל-{fg_now} היום")
    elif fg_now is not None and fg_month is not None and abs(fg_now - fg_month) > 20:
        direction = 'ירד' if fg_now < fg_month else 'עלה'
        market_signals.append(f"סנטימנט השוק {direction} חדות מאז החודש שעבר — מ-{fg_month} ל-{fg_now}")

    # ── Sector ETF map for stock-vs-sector comparison ──
    _SECTOR_TO_ETF = {
        'Technology': 'טק', 'Communication Services': 'טק',
        'Financial Services': 'פיננסים', 'Financial': 'פיננסים',
        'Healthcare': 'בריאות',
        'Energy': 'אנרגיה',
    }
    sector_etf_moves = macro.get('sector_etfs', {})

    # ── TA-35 for Israeli stocks ──
    ta35 = macro.get('ta35')
    ta35_change = ta35.get('change_pct', 0) if isinstance(ta35, dict) else 0

    for s in valid_stocks:
        ticker = s['ticker']
        signals = []
        change = s.get('change_pct', 0)
        hebrew = s.get('hebrew', ticker)

        # Signal 1: Stock vs Sector relative performance
        sector = s.get('sector', '')
        etf_label = _SECTOR_TO_ETF.get(sector)
        if etf_label and etf_label in sector_etf_moves:
            sector_change = sector_etf_moves[etf_label]
            relative = change - sector_change
            if abs(relative) > 1.0:
                perf = 'ביצוע יתר' if relative > 0 else 'ביצוע חסר'
                signals.append(f"{hebrew} {change:+.1f}% לעומת הסקטור ({etf_label}) {sector_change:+.1f}% — {perf} של {abs(relative):.1f}%")

        # Signal 2: Valuation direction (forward PE vs trailing PE)
        pe = s.get('pe')
        fwd_pe = s.get('forward_pe')
        if pe and fwd_pe and pe > 0 and fwd_pe > 0:
            if fwd_pe < pe * 0.85:
                signals.append(f"מכפיל עתידי ({fwd_pe:.0f}) נמוך משמעותית מנוכחי ({pe:.0f}) — השוק מצפה לצמיחת רווחים")
            elif fwd_pe > pe * 1.15:
                signals.append(f"מכפיל עתידי ({fwd_pe:.0f}) גבוה מנוכחי ({pe:.0f}) — השוק מצפה להאטה")

        # Signal 4: Analyst target spread (conviction)
        t_low = s.get('target_low')
        t_high = s.get('target_high')
        t_mean = s.get('target_mean')
        if t_low and t_high and t_mean and t_mean > 0:
            spread_pct = (t_high - t_low) / t_mean * 100
            if spread_pct > 60:
                signals.append(f"פיזור יעדים רחב מאוד (${t_low:.0f}-${t_high:.0f}) — חוסר הסכמה חד בין אנליסטים")
            elif spread_pct < 20:
                signals.append(f"קונצנזוס אנליסטים חזק — יעדים בטווח צר (${t_low:.0f}-${t_high:.0f})")
            # Also flag if price above target
            price = s.get('price', 0)
            if price and t_mean and price > t_mean * 1.05:
                signals.append(f"המחיר (${price:.0f}) מעל יעד הממוצע (${t_mean:.0f}) — האנליסטים חושבים שהמניה יקרה")

        # Signal 5: "Moving with the market" detection
        if abs(change - sp500_change) < 0.5 and abs(change) < 1.5:
            signals.append(f"זזה עם השוק ({change:+.1f}% לעומת S&P {sp500_change:+.1f}%) — אין סיפור ספציפי")

        # Signal 6: Earnings countdown
        if s.get('earnings_date'):
            try:
                from datetime import date as _d
                ed = datetime.strptime(s['earnings_date'], '%Y-%m-%d').date()
                days = (ed - _d.today()).days
                if days == 0:
                    signals.append("דוח רווחים היום! צפי לתנודתיות חריגה")
                elif 1 <= days <= 7:
                    eps_note = f" (צפי EPS: ${s['earnings_est_eps']})" if s.get('earnings_est_eps') else ''
                    signals.append(f"דוח רווחים בעוד {days} ימים{eps_note} — אזור תנודתיות גבוהה")
                elif 8 <= days <= 14:
                    signals.append(f"דוח רווחים בעוד {days} יום — סוחרים מתחילים להתמקם")
            except Exception:
                pass

        # Signal 7: Israeli stock vs TA-35
        resolved = s.get('resolved', ticker)
        if resolved.endswith('.TA') and ta35_change:
            relative = change - ta35_change
            if abs(relative) > 1.0:
                perf = 'ביצוע יתר' if relative > 0 else 'ביצוע חסר'
                signals.append(f"{hebrew} {change:+.1f}% לעומת ת\"א 35 {ta35_change:+.1f}% — {perf}")

        # Cap at 3 signals per stock (priority: order added)
        stock_signals[ticker] = signals[:3]

    # ── Signal 8: Dominant macro event detection ──
    # When geopolitical / macro events are THE story, flag it so the LLM leads with macro
    macro_news = macro.get('macro_news') or []
    _GEO_KWS = ['war', 'iran', 'strait', 'hormuz', 'missile', 'attack', 'invasion',
                 'sanctions', 'tariff', 'embargo', 'ceasefire', 'conflict', 'nuclear',
                 'nato', 'china', 'taiwan', 'opec', 'oil crisis', 'recession',
                 'crash', 'bank run', 'default', 'collapse']
    geo_stories = []
    for story in macro_news:
        title_low = story['title'].lower()
        matched = [kw for kw in _GEO_KWS if kw in title_low]
        if len(matched) >= 1 and story.get('score', 0) >= 4:
            geo_stories.append(story['title'])

    dominant_macro = None
    vix = macro.get('vix', 0) or 0
    if len(geo_stories) >= 2 or (len(geo_stories) >= 1 and vix >= 25):
        # Multiple geopolitical stories or geo + high fear = macro is THE story
        dominant_macro = '\n'.join(f"• {s}" for s in geo_stories[:4])
        market_signals.insert(0,
            f"⚠️ אירוע מאקרו/גיאופוליטי דומיננטי — זה הסיפור המרכזי של היום, לא מניה ספציפית!"
        )

    return {
        'market_signals': market_signals,
        'stock_signals': stock_signals,
        'dominant_macro': dominant_macro,
    }


def _generate_podcast_with_llm(data, user_name=None):
    """Full LLM podcast script — Planet Money meets All-In structure.

    Techniques applied (from analysis of top financial podcasts):
    - Cold open with Zeigarnik loop: drop mid-scene, raise a question, close it at the end
    - Stakes ladder: personal portfolio → sector → market-wide → back to personal
    - Camera pull-back beat: one small detail that reveals large systemic implications
    - Open loop tease before the main story, resolved at closing
    - False consensus / consequence reversal framing
    - Personal stakes injection: every macro data point connects to the portfolio
    - Expert voices: analyst upgrades/downgrades, Reddit signal, sentiment data
    """
    from datetime import date as _date

    m = data['macro']
    stocks = data['stocks']
    valid_stocks = [s for s in stocks if 'error' not in s]
    _dt = datetime.fromisoformat(data['date'])
    date_str = _dt.strftime('%d/%m/%Y')
    _day_names = ['שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת', 'ראשון']
    date_str_full = f"יום {_day_names[_dt.weekday()]}, {date_str}"

    # Sort by interest score — combines price move, earnings proximity, news volume, and market cap
    def _interest_score(s):
        score = abs(s.get('change_pct', 0)) * 3
        score += min(s.get('news_summary', {}).get('article_count', 0) * 0.4, 8)
        # Earnings proximity bonus
        if s.get('earnings_date'):
            try:
                from datetime import date as _d2
                days = (datetime.strptime(s['earnings_date'], '%Y-%m-%d').date() - _d2.today()).days
                if 1 <= days <= 7:
                    score += 25
                elif 1 <= days <= 14:
                    score += 18
                elif 1 <= days <= 30:
                    score += 10
            except Exception:
                pass
        # Large-cap floor — AMZN/META/MSFT etc. never buried
        mcap = s.get('market_cap') or 0
        if mcap > 500e9:
            score = max(score, 18)
        elif mcap > 100e9:
            score = max(score, 12)
        return score

    valid_stocks_sorted = sorted(valid_stocks, key=_interest_score, reverse=True)

    # ── Portfolio P&L (set by generate_text_report before this call) ──
    portfolio_stocks = [s for s in valid_stocks if '_dollar_impact' in s]
    portfolio_ctx = ''
    if portfolio_stocks:
        total_pnl   = sum(s['_dollar_impact'] for s in portfolio_stocks)
        total_value = sum(s.get('_shares', 0) * s['price'] for s in portfolio_stocks)
        pct = (total_pnl / total_value * 100) if total_value else 0
        sign = '+' if total_pnl >= 0 else ''
        by_impact = sorted(portfolio_stocks, key=lambda s: s['_dollar_impact'])
        losers  = [s for s in by_impact if s['_dollar_impact'] < 0]
        winners = [s for s in by_impact if s['_dollar_impact'] > 0]
        pnl_exact = f"{sign}${abs(total_pnl):,.0f}"
        # Keep daily change and unrealized P&L clearly separated so LLM doesn't mix them up
        portfolio_ctx = (
            f"=== נתוני תיק ===\n"
            f"שווי תיק היום: ${total_value:,.0f}\n"
            f"שינוי ב-24 שעות: {pnl_exact} בלבד — זה {sign}{pct:.2f}% מהשווי, לא יותר\n"
        )
        if losers:
            portfolio_ctx += f"הכי הוריד: {losers[0].get('hebrew', losers[0]['ticker'])} ({losers[0]['_dollar_impact']:+,.0f}$, {losers[0]['change_pct']:+.1f}% היום)\n"
        if winners:
            portfolio_ctx += f"הכי הרים: {winners[-1].get('hebrew', winners[-1]['ticker'])} ({winners[-1]['_dollar_impact']:+,.0f}$, {winners[-1]['change_pct']:+.1f}% היום)\n"
        # Total unrealized P&L — clearly labelled as since-purchase, not today
        unrealized_stocks = [s for s in portfolio_stocks if '_unrealized_pnl' in s]
        if unrealized_stocks:
            total_unrealized = sum(s['_unrealized_pnl'] for s in unrealized_stocks)
            total_cost = sum(s['_cost_basis'] for s in unrealized_stocks)
            unr_sign = '+' if total_unrealized >= 0 else ''
            unr_pct = (total_unrealized / total_cost * 100) if total_cost else 0
            portfolio_ctx += (
                f"רווח כולל מאז הקנייה (לא ממומש, לא שינוי היום): "
                f"{unr_sign}${total_unrealized:,.0f} ({unr_sign}{unr_pct:.1f}% על ההשקעה המקורית)\n"
            )
        portfolio_ctx += "=== סוף נתוני תיק ===\n"

    # ── Macro ──
    fg = m.get('fear_greed', {})
    sp = m.get('sp500', {})
    macro_ctx = (
        f"S&P 500: {sp.get('price','?')} ({sp.get('change_pct',0):+.2f}%)\n"
        f"VIX: {m.get('vix','?')}\n"
        f"Fear & Greed: {fg.get('score','?')} ({fg.get('rating','')}) — שבוע קודם: {fg.get('week_ago','?')}, חודש קודם: {fg.get('month_ago','?')}\n"
        f"USD/ILS: ₪{m.get('usd_ils','?')}\n"
    )
    if sp.get('rsi'):
        rsi_label = 'Oversold — שוק ירוד' if sp['rsi'] < 30 else 'Overbought — שוק חם' if sp['rsi'] > 70 else 'Neutral'
        macro_ctx += f"RSI S&P: {sp['rsi']} ({rsi_label})\n"

    # Macro/geopolitical news stories moving markets today
    macro_news = m.get('macro_news') or []
    if macro_news:
        macro_ctx += "\n=== חדשות מאקרו קריטיות שמניעות את השוק היום ===\n"
        _html_re = re.compile(r'<[^>]*>?')  # handles both complete and truncated tags
        for story in macro_news:
            title = _html_re.sub('', story['title']).strip()
            macro_ctx += f"• {title}\n"
            raw_snip = story.get('snippet', '')
            # Skip snippet if it looks like raw HTML (starts with <)
            if raw_snip and not raw_snip.lstrip().startswith('<'):
                snippet = _html_re.sub('', raw_snip).strip()
                if snippet and snippet != title:
                    macro_ctx += f"  {snippet[:220]}\n"
        macro_ctx += "=== סוף חדשות מאקרו ===\n"

    def _clean_expert_quote(text):
        """Return clean tweet text, or None if content looks like metadata/garbage."""
        if not text or len(text) < 60:
            return None
        # Reject profile boilerplate and metadata
        boilerplate = ['profile.', 'reposted by', 'followers', 'following', 'joined ',
                       'image on x', 'javascript is not available', 'official x account',
                       'sep ', 'oct ', 'nov ', 'dec ', 'jan ', 'feb ', 'mar ', 'apr ']
        low = text.lower()
        if any(b in low for b in boilerplate):
            return None
        # Reject if too many standalone numbers (engagement counts like "90 328 2932")
        words = text.split()
        num_count = sum(1 for w in words if re.match(r'^\d[\d,\.]*$', w))
        if len(words) > 0 and num_count / len(words) > 0.25:
            return None
        # Take the most substantive sentence (longest, starts with capital/uppercase)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?·])\s+', text) if len(s.strip()) > 50]
        best = max(sentences, key=len) if sentences else text
        return best[:200]

    # ── Per-stock context blocks ──
    stock_blocks = []
    for s in valid_stocks_sorted:
        b = [f"=== {s.get('hebrew', s['ticker'])} ({s['ticker']}) ==="]
        b.append(f"מחיר: {s.get('currency','$')}{s.get('price','?')} | שינוי: {s.get('change_pct',0):+.1f}%")

        if '_dollar_impact' in s:
            b.append(f"השפעה על התיק: {s['_dollar_impact']:+,.0f}$ ({s.get('_shares',0):g} מניות)")
        if '_entry_price' in s and '_unrealized_pnl' in s:
            unr_sign = '+' if s['_unrealized_pnl'] >= 0 else ''
            b.append(f"עלות בסיס: ${s['_entry_price']:,.2f} | רווח/הפסד כולל: {unr_sign}${s['_unrealized_pnl']:,.0f} ({unr_sign}{s['_unrealized_pct']:.1f}%)")

        # Price context — time horizon makes prices meaningful
        if s.get('change_1y') is not None:
            b.append(f"שנה אחורה: {s['change_1y']:+.1f}% | היום: {s.get('currency','$')}{s.get('price','?')}")
        if s.get('w52_high') and s.get('w52_low') and s.get('price'):
            range_pos = (s['price'] - s['w52_low']) / (s['w52_high'] - s['w52_low']) * 100 if s['w52_high'] != s['w52_low'] else 50
            b.append(f"טווח 52 שבוע: ${s['w52_low']} – ${s['w52_high']} | עכשיו {range_pos:.0f}% מהתחתית")

        # Technical signals — only if significant
        if s.get('rsi'):
            if s['rsi'] < 33:
                b.append(f"RSI: {s['rsi']} — מכר יתר חריג. אזור היסטורי לריבאונד.")
            elif s['rsi'] > 67:
                b.append(f"RSI: {s['rsi']} — קנית יתר. לחץ מכירה אפשרי.")
        if s.get('ma200'):
            pct_ma = round((s['price'] - s['ma200']) / s['ma200'] * 100, 1)
            if abs(pct_ma) > 10:
                b.append(f"MA200: {'מעל' if pct_ma > 0 else 'מתחת'} ב-{abs(pct_ma):.1f}% — {'מגמה עולה חזקה' if pct_ma > 0 else 'מתחת לממוצע ארוך טווח'}")
        if s.get('pct_from_high') and s['pct_from_high'] < -30:
            b.append(f"מרחק מ-52W High: {s['pct_from_high']:+.1f}% — ירידה משמעותית מהשיא")

        # Fundamentals — valuation & growth
        fund_parts = []
        if s.get('pe') and s['pe'] > 0:
            fund_parts.append(f"P/E: {s['pe']:.1f}")
        if s.get('forward_pe') and s['forward_pe'] > 0:
            fund_parts.append(f"P/E צפוי: {s['forward_pe']:.1f}")
        if s.get('revenue_growth') and abs(s['revenue_growth']) > 1:
            fund_parts.append(f"צמיחת הכנסות: {s['revenue_growth']:+.0f}%")
        if s.get('earnings_growth') and abs(s['earnings_growth']) > 1:
            fund_parts.append(f"צמיחת רווח: {s['earnings_growth']:+.0f}%")
        if s.get('profit_margin') and s['profit_margin'] > 0:
            fund_parts.append(f"מרווח רווחי: {s['profit_margin']:.0%}")
        if fund_parts:
            b.append(' | '.join(fund_parts))

        # Analyst consensus
        if s.get('target_mean'):
            upside = round((s['target_mean'] - s['price']) / s['price'] * 100, 1)
            b.append(f"קונצנזוס אנליסטים: יעד {s.get('currency','$')}{s['target_mean']} ({upside:+.1f}% פוטנציאל) | {s.get('recommendation','?')}")
        # Latest analyst action — a named analyst or firm took a position
        actions = s.get('recent_analyst_actions', [])
        if actions:
            a = actions[0]
            label_map = {'up': 'שדרג', 'down': 'הוריד', 'init': 'פתח כיסוי', 'reit': 'שמר', 'main': 'שמר'}
            verb = label_map.get(a['action'].lower(), a['action'])
            grade = f" ל-{a['to_grade']}" if a.get('to_grade') else ''
            pt = f" עם יעד ${a['price_target']:,.0f}" if a.get('price_target') else ''
            b.append(f"דירוג: {a['firm']} {verb}{grade}{pt} ({a['date']})")

        # Short interest — squeeze or confirmation signal
        if s.get('short_pct') and s['short_pct'] > 5:
            squeeze_note = ' — פוטנציאל לסחיטת שורטים (short squeeze)' if s['short_pct'] > 15 else ''
            days_cover = f" | {s['short_ratio']} ימי כיסוי" if s.get('short_ratio') else ''
            b.append(f"שורט: {s['short_pct']:.1f}% מהמניות הנסחרות{days_cover}{squeeze_note}")

        # Insider activity — smart money signal
        insiders = s.get('insider_activity', [])
        if insiders:
            # Summarise: total bought vs sold in 90 days
            buys  = [x for x in insiders if x['type'] == 'קנייה']
            sells = [x for x in insiders if x['type'] == 'מכירה']
            if sells:
                top_sell = max(sells, key=lambda x: x['value'])
                b.append(
                    f"פנים — מכירה: {top_sell['name']} ({top_sell['relation']}) "
                    f"מכר ${top_sell['value']:,} ({top_sell['date']})"
                    + (f" + {len(sells)-1} נוספים" if len(sells) > 1 else "")
                )
            if buys:
                top_buy = max(buys, key=lambda x: x['value'])
                b.append(
                    f"פנים — קנייה: {top_buy['name']} ({top_buy['relation']}) "
                    f"קנה ${top_buy['value']:,} ({top_buy['date']})"
                )

        # Upcoming earnings — forward-looking hook
        if s.get('earnings_date'):
            try:
                ed = datetime.strptime(s['earnings_date'], '%Y-%m-%d').date()
                days = (ed - _date.today()).days
                if 1 <= days <= 45:
                    eps = f" | EPS צפוי: ${s['earnings_est_eps']}" if s.get('earnings_est_eps') else ''
                    b.append(f"דוח רווחים: בעוד {days} יום ({ed.strftime('%d/%m/%Y')}){eps}")
                elif days == 0:
                    b.append("דוח רווחים: היום!")
            except Exception:
                pass

        # News — raw headlines only (avoid summary-of-summary problem)
        raw_articles = s.get('news_deep', [])
        if raw_articles:
            headlines = []
            for a in raw_articles[:6]:
                title = (a.get('title') or '').strip()
                snippet = (a.get('content') or a.get('summary') or '').strip()[:180]
                if title:
                    entry = f"  • {title}"
                    if snippet and snippet.lower() != title.lower():
                        entry += f" — {snippet}"
                    headlines.append(entry)
            if headlines:
                b.append("כותרות גולמיות (השתמש בהן ישירות בניתוח):\n" + "\n".join(headlines))

        # Social intelligence
        social = s.get('social', {})
        reddit = social.get('reddit', [])
        if reddit:
            top = reddit[0]
            b.append(f"Reddit (הכי פופולרי, ↑{top.get('ups',0):,}): \"{top['title'][:130]}\"")
            if len(reddit) > 1:
                b.append(f"Reddit (נוסף): \"{reddit[1]['title'][:100]}\"")
        sent = social.get('sentiment')
        if sent and sent.get('total_messages', 0) >= 10:
            bull, bear, total = sent['bullish_pct'], sent['bearish_pct'], sent['total_messages']
            sentiment_note = ''
            if bull >= 70:
                sentiment_note = f'שורי מאוד ({bull}%) — {total} הודעות'
            elif bear >= 60:
                sentiment_note = f'דובי ({bear}%) — {total} הודעות'
            elif bull >= 55:
                sentiment_note = f'נטייה שורית ({bull}%) — {total} הודעות'
            if sentiment_note:
                b.append(f"StockTwits: {sentiment_note}")
        # Viral tweets (Twitter API v2) — sorted by engagement score
        viral = social.get('viral_tweets', [])
        for vt in viral[:3]:
            tag = '⭐ מומחה' if vt.get('is_expert') else f"🔥 {vt.get('likes',0):,} לייקים"
            b.append(f"X [{tag}] {vt.get('author','')}: \"{vt['text'][:220]}\"")
        # Expert X posts via Tavily (fallback when no viral tweets) — filtered for relevance
        if not viral:
            _sticker = s['ticker']
            _fin_keywords = [
                'stock', 'share', 'market', 'price', 'earnings', 'revenue', 'profit',
                'buy', 'sell', 'invest', 'rally', 'drop', 'trade', 'analyst', 'rating',
                'quarter', 'growth', 'margin', 'valuation', 'bull', 'bear', 'rate',
                _sticker.lower(), (s.get('name', '') or '').lower().split()[0],
            ]
            for ep in social.get('twitter', [])[:3]:
                clean = _clean_expert_quote(ep.get('text', ''))
                if not clean:
                    continue
                # Must contain at least one financial keyword — reject off-topic content
                low = clean.lower()
                if not any(kw in low for kw in _fin_keywords if kw):
                    continue
                b.append(f"X / {ep.get('author','מומחה')}: \"{clean}\"")

        stock_blocks.append('\n'.join(b))

    # ── Detect named expert voices for injection ──
    expert_quotes = []
    seen_tickers = set()
    for s in valid_stocks_sorted:
        ticker = s['ticker']
        if ticker in seen_tickers:
            continue
        for ep in s.get('social', {}).get('twitter', [])[:2]:
            author = ep.get('author', '').lstrip('@')
            clean = _clean_expert_quote(ep.get('text', ''))
            if clean and author:
                expert_quotes.append(f"@{author} על {s.get('hebrew', ticker)}: \"{clean}\"")
                seen_tickers.add(ticker)
                break
    expert_ctx = '\n'.join(expert_quotes[:3]) if expert_quotes else ''

    # ── Detect transformation narrative ──
    transformation_stocks = []
    for s in valid_stocks_sorted:
        if (s.get('pct_from_high') or 0) < -35:
            news_text = s.get('news_summary', {}).get('summary', '').lower()
            if any(w in news_text for w in ['ai', 'artificial intelligence', 'pivot', 'transform', 'new strategy', 'rebrand']):
                transformation_stocks.append(
                    f"{s.get('hebrew', s['ticker'])}: חברה בטרנספורמציה — {abs(s['pct_from_high']):.0f}% מהשיא, אבל הסיפור הוא מה שהיא הופכת להיות"
                )
    transformation_ctx = '\n'.join(transformation_stocks) if transformation_stocks else ''

    # ── Sector grouping — find unified narratives ──
    _SECTOR_HEB = {
        'Technology': 'טק', 'Communication Services': 'תקשורת/טק',
        'Consumer Cyclical': 'צרכנות', 'Consumer Defensive': 'צרכנות בסיסית',
        'Financial Services': 'פיננסים', 'Healthcare': 'בריאות',
        'Energy': 'אנרגיה', 'Basic Materials': 'חומרי גלם',
        'Industrials': 'תעשייה', 'Real Estate': 'נדל"ן', 'Utilities': 'שירותים',
    }
    from collections import defaultdict
    sector_buckets = defaultdict(list)
    for s in valid_stocks_sorted:
        sec = _SECTOR_HEB.get(s.get('sector', ''), s.get('sector', 'אחר') or 'אחר')
        sector_buckets[sec].append(s)

    # Only sectors with 2+ stocks — the unified narrative candidates
    sector_ctx_lines = []
    for sec, stocks_in_sec in sector_buckets.items():
        if len(stocks_in_sec) < 2:
            continue
        names = ', '.join(s.get('hebrew', s['ticker']) for s in stocks_in_sec)
        avg_chg = sum(s.get('change_pct', 0) for s in stocks_in_sec) / len(stocks_in_sec)
        direction = 'ירד' if avg_chg < 0 else 'עלה'
        sector_ctx_lines.append(
            f"סקטור {sec}: {names} — ממוצע {direction} {abs(avg_chg):.1f}% ← ייתכן שסיפור סקטוריאלי אחד מסביר את כולם"
        )
    sector_ctx = '\n'.join(sector_ctx_lines) if sector_ctx_lines else ''

    # ── Portfolio concentration ──
    concentration_ctx = ''
    if portfolio_stocks:
        sec_val = defaultdict(float)
        for s in portfolio_stocks:
            sec = _SECTOR_HEB.get(s.get('sector', ''), s.get('sector', 'אחר') or 'אחר')
            sec_val[sec] += s.get('_shares', 0) * s.get('price', 0)
        total_v = sum(sec_val.values())
        if total_v > 0:
            top_sectors = sorted(sec_val.items(), key=lambda x: -x[1])[:4]
            parts = [f"{sec} {v/total_v*100:.0f}%" for sec, v in top_sectors]
            concentration_ctx = f"ריכוז סקטוריאלי בתיק: {', '.join(parts)}"

    # ── Macro: TNX + sector ETFs context string ──
    tnx = m.get('treasury_10y')
    tnx_ctx = ''
    if tnx and tnx.get('yield'):
        bps = tnx['change_bps']
        direction = 'עלה' if bps > 0 else 'ירד'
        impact = ' — לחץ על מכפילי הטק' if bps > 5 else (' — גב רוח לצמיחה' if bps < -5 else '')
        tnx_ctx = f"תשואת אג\"ח 10 שנים (ארה\"ב): {tnx['yield']:.2f}% ({direction} {abs(bps):.0f} נקודות בסיס){impact}"

    sector_etf_ctx = ''
    if m.get('sector_etfs'):
        etf_parts = [f"{label} {chg:+.1f}%" for label, chg in m['sector_etfs'].items()]
        sector_etf_ctx = f"תנועות סקטוריאליות: {', '.join(etf_parts)}"

    # ── Assemble full context for LLM ──
    macro_full = macro_ctx
    if tnx_ctx:
        macro_full += f"{tnx_ctx}\n"
    if sector_etf_ctx:
        macro_full += f"{sector_etf_ctx}\n"

    context_parts = [f"תאריך: {date_str_full}", f"\nמאקרו:\n{macro_full}"]
    if portfolio_ctx:
        context_parts.append(f"תיק ההשקעות:\n{portfolio_ctx}")
    if concentration_ctx:
        context_parts.append(concentration_ctx)
    # ── Compute analytical signals ──
    analytical = _compute_analytical_signals(valid_stocks_sorted, m)

    # Inject per-stock signals into stock blocks
    for i, s in enumerate(valid_stocks_sorted):
        ticker_signals = analytical['stock_signals'].get(s['ticker'], [])
        if ticker_signals:
            stock_blocks[i] += "\n→ תובנות:\n" + "\n".join(f"  → {sig}" for sig in ticker_signals)

    context_parts.append("מניות (מהגדול לקטן לפי תנועה):\n" + "\n\n".join(stock_blocks))

    # Market-level analytical signals
    if analytical['market_signals']:
        context_parts.append(
            "=== תובנות שוק (חובה להשתמש!) ===\n" +
            "\n".join(f"→ {sig}" for sig in analytical['market_signals'])
        )

    if sector_ctx:
        context_parts.append(f"ניתוח סקטוריאלי — חפש נרטיב מאחד:\n{sector_ctx}")
    if expert_ctx:
        context_parts.append(f"ציטוטים ממומחים (X/Twitter) — חייב להשתמש בהם בשמם:\n{expert_ctx}")
    if transformation_ctx:
        context_parts.append(f"נרטיב טרנספורמציה — השתמש במסגור זה:\n{transformation_ctx}")
    full_context = "\n\n".join(context_parts)

    has_portfolio = bool(portfolio_ctx)
    all_names_heb = ', '.join(s.get('hebrew', s['ticker']) for s in valid_stocks_sorted)

    _user_greeting = f"היום ננתח את התיק של {user_name}." if user_name else ""

    # ── Emotional tone guidance ──
    portfolio_dollar = None
    if has_portfolio:
        try:
            total_change = sum(s.get('_dollar_impact', 0) for s in valid_stocks_sorted)
            portfolio_dollar = total_change
        except Exception:
            pass

    if portfolio_dollar is not None and portfolio_dollar < -500:
        mood = "התיק ירד. פתח ברוגע ובכנות — אל תמעיט, אל תבעית."
    elif portfolio_dollar is not None and portfolio_dollar > 500:
        mood = "יום טוב בתיק. אפשר להיות שמח אבל לא נלהב יתר על המידה."
    else:
        mood = "יום בלי תנועות גדולות. אל תנפח ואל תייצר דרמה — פשוט ספר מה קרה."

    # ── Determine main story type ──
    dominant_macro = analytical.get('dominant_macro')
    if dominant_macro:
        main_story_instruction = (
            "4. הסיפור המרכזי — האירוע הגיאופוליטי/מאקרו שמניע את השוק, 60-70% מהפרק.\n"
            "   הכותרות:\n"
            f"{dominant_macro}\n"
            "   ספר מה קורה → למה זה משנה למשקיע → איך זה משפיע על המניות בתיק שלך.\n"
            "   חבר את האירוע למניות ספציפיות: מי נפגע, מי מרוויח, מי בסיכון.\n"
            "   אל תסתפק ב'אי-ודאות גיאופוליטית' — תגיד בדיוק מה קורה ומה המשמעות."
        )
    else:
        main_story_instruction = (
            "4. הסיפור המרכזי — מניה אחת או נושא אחד, 60-70% מהפרק.\n"
            "   ספר את הסיפור: מה קרה → למה → מה זה אומר לתיק.\n"
            "   השתמש בנתוני הערכת שווי (PE, צמיחת הכנסות) אם יש — הם מסבירים \"למה\"."
        )

    # ── THE PROMPT ──
    prompt = f"""{full_context}

{'='*60}

=== הנחיות ===

אתה מגיש את "אסטרה פייננס" — פודקאסט יומי קצר שמספר למשקיע מה קרה עם הכסף שלו.
הטון: כמו הודעת קולית ארוכה לחבר שמבין קצת מניות. לא דוח, לא כתבה, לא מצגת.

טון רגשי להיום: {mood}

=== מבנה הפרק (בסדר הזה) ===

1. פתיחה מותגית (משפט אחד בלבד):
   "ברוכים הבאים לפודקאסט התיק האישי שלך, אסטרה פייננס. {_user_greeting}"

2. תאריך + disclaimer (משפט אחד בלבד):
   "[תאריך היום בעברית]. תזכורת — מה שנאמר כאן אינו ייעוץ השקעות, אלא סיכום חדשות ומידע בלבד."

3. כותרת התיק ({f"התיק שלך עלה/ירד ב-{abs(portfolio_dollar):,.0f} דולר היום." if portfolio_dollar is not None else "אמור בקצרה איך התיק נראה היום."})

{main_story_instruction}

5. סיכום מהיר — מקסימום 3 מניות נוספות, משפט אחד כל אחת. רק אם יש סיבה אמיתית.
   אם מניה עשתה אפס ואין לה חדשות — אל תזכיר אותה בכלל.
   אל תחזור על מניות שכבר הוזכרו בסיפור המרכזי.

6. סגירה: "זהו להיום. תודה שהאזנת, ונשמע שוב מחר."

=== כללי כתיבה ===

מספרים:
• עגל: "בערך 550 דולר" ולא "547 דולר ו-23 סנט". "כמעט שלושים אחוז" ולא "29.7 אחוז"
• דולר ואחוז — כתוב כספרות ($550, 30%). ה-TTS יטפל בהקראה
• כל אחוז חייב הקשר: לא "פוטנציאל של 32%" — אלא "האנליסטים חושבים שהמניה שווה 32% יותר מהמחיר היום"

שמות:
• חברות בעברית בלבד: {all_names_heb}
• כל שם של בנק/חברת ניתוח חייב הסבר: "ברקלייז, שמכסה את המניה" — לא סתם "ברקלייז"

סיפור סקטוריאלי, לא רשימת מניות:
• אל תעבור מניה-מניה עם אחוזים ("סופי עלתה 2.6%, אמזון עלתה 1.4%, מטא ירדה 0.2%"). זה משעמם.
• במקום: קבץ לפי סקטור או נרטיב. "מניות הטכנולוגיה בתיק ירדו קלות, הבולטת היא פאלו אלטו שהפסידה כמעט אחוז."
• הזכר שם מניה רק כשיש לה סיפור ספציפי — לא כדי לדווח על אחוז.
• מותר להזכיר 4-5 מניות בסך הכל בכל הפרק. השאר — קבץ לסקטור.

טון:
• גוף שני: "שלך", "התיק שלך", "אתה"
• הכנס 2-3 סמני שיחה טבעיים: "תקשיב", "רגע", "אז ככה", "בוא נגיד"
• אם אין מה לומר על מניה — "אין חדשות מהותיות" ותעבור הלאה. עדיף משפט קצר וכנה מפסקה ריקה
• שחרר רגשות: "זה מספר יפה", "זה קצת מדאיג", "אני חושב שזה מעניין"

כלל ברזל — תמיד עובדות קונקרטיות:
• לא "התיק נראה רגוע / יציב / די יציב" — אלא "התיק עלה ב-$200 היום" או "התיק כמעט לא זז"
• לא "המניה במצב מעניין" — אלא "המניה עלתה 3% אחרי דוח רווחים"
• לא "הסקטור יציב" — אלא "אמזון ומרקדו ליברה עלו באחוז, בלי חדשות מיוחדות"
• אל תתאר — ספר. כל משפט צריך לכלול עובדה, מספר, או אירוע. אם אין — אל תגיד כלום

חזרות:
• אל תזכיר את אותה מניה יותר מפעמיים בכל הפרק. אם כבר דיברת עליה — אל תחזור.
• אל תחזור על אותו מספר/אחוז פעמיים. אם אמרת "סופי עלתה 2.6%" — לא צריך לחזור על זה.

ביטויים אסורים — כל ביטוי שלא היית אומר בהודעת וואטסאפ:
✗ "ניכר כי", "ראוי לציין", "יש לציין", "בהתאם לכך", "הנתונים מצביעים"
✗ "מתי ההבטחה הופכת לרווח", "פער בין הסיפור לביצוע"
✗ "שוק שמנסה להבחין", "תמונה גדולה", "המנגנון הבסיסי"
✗ "מן הראוי", "נראה כי", "בד בבד", "לצד זאת"
✗ "נראה די יציב", "די יציב", "נראה רגוע", "מורכבות", "דינמיקה", "בריטארטיות"
✗ כל ביטוי שנשמע כמו כותרת עיתון, מאמר אקדמי, או דוח אנליסטים
✗ אל תמציא מילים. אם אתה לא בטוח שמילה קיימת בעברית — השתמש במילה פשוטה

כתיב — חשוב מאוד:
• אתה כותב בעברית לדובר עברית שפת אם. כל ביטוי שבור ישמע מוזר. אם אתה לא בטוח — כתוב משפט קצר ופשוט.
• אל תרכיב ביטויים חדשים. השתמש רק בביטויים שאתה בטוח שקיימים בעברית.
• דוגמאות לביטויים שבורים שאסור לכתוב:
  ✗ "מורידים ספקות" → ✓ "יורדים"
  ✗ "למתחת לקו דעתם של המשקיעים" → ✓ "מתחת לרדאר של המשקיעים" או פשוט "המשקיעים לא מודאגים"
  ✗ "הבונוסים שהבנק חטף" → ✓ "הרווחים הגבוהים של הבנק"
  ✗ "מכפילים ממזערים ציפיות" → ✓ "השוק מצפה לפחות צמיחה"
  ✗ "תיק הרווחים הרשמי" → ✓ "עונת הדוחות"
  ✗ "מתקפלות" (בהקשר של חברות) → ✓ "מתפתחות" או "הולכות"
• כלל אצבע: אם המשפט נשמע כמו תרגום מאנגלית — כתוב אותו מחדש בעברית טבעית.

פורמט:
• ללא markdown, ללא כותרות, ללא קווים — טקסט רציף בלבד
• שבור שורה אחרי כל 2-3 משפטים — זה יוצר הפסקות טבעיות באודיו
• אורך: 400-600 מילה (כ-4 דקות האזנה)

=== חובת ניתוח ===

כשיש שורות שמתחילות ב-→ (תובנות) — חייב להשתמש בהן. הן הניתוח שלך. אל תתעלם מהן.
הן עונות על "למה" ו"מה זה אומר" — תשלב אותן בסיפור כאילו הן החשיבה שלך.

בפרק, ענה על 3 השאלות האלה (לא כרשימה — בתוך הזרימה הטבעית):
1. מה הדבר הכי מפתיע היום בתיק? (לא הכי גדול — הכי מפתיע)
2. למה זה קרה? (חבר נקודות: מאקרו, סקטור, חדשות)
3. מה לשים לב אליו השבוע? (דוח קרוב, מגמה, סיכון)

דוגמה לשימוש בתובנה:
• נתון: "סופי עלתה 2.6%"
• תובנה: "→ ביצוע יתר: סופי +2.6% לעומת סקטור הפיננסים +1.0%"
• מה לומר: "תקשיב, סופי עלתה 2.6% — אבל מה שמעניין זה שהפיננסים ביחד עלו רק אחוז. סופי רצה לבד, וזה בגלל..."

אל תתאר — נתח. כל משפט צריך לענות על "אז מה?" ולא רק על "מה קרה?"."""

    script = _llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=4000,
        system=(
            "אתה אנליסט, לא קריין. התפקיד שלך לנתח — לא לתאר.\n"
            "כשאתה רואה שורות → (תובנות) — זה הניתוח שלך. תשלב אותן בסיפור.\n"
            "אתה מדבר כמו חבר ישראלי שמבין פיננסים — לא כלכלן, לא עיתונאי.\n"
            "דוגמה לטון נכון: 'תקשיב, סופי עלתה 2.6% אבל הפיננסים עלו רק אחוז — היא רצה לבד'\n"
            "דוגמה לטון שגוי: 'ניכר כי סופי מציגה ביצועי יתר ביחס לסקטור'\n"
            "כל מספר — עם הקשר. כל שם — עם הסבר. אם אין מה לומר — אמור שאין ותעבור הלאה.\n"
            "עברית של שיחה — לא עברית של כתבה. אל תמציא ביטויים — אם אתה לא בטוח, כתוב משפט פשוט וקצר.\n"
            "אל תתרגם ביטויים מאנגלית מילה במילה. כתוב כמו שישראלי מדבר."
        )
    )

    # ── Post-processing ──
    for ch in ('**', '*', '## ', '### ', '# ', '[', ']'):
        script = script.replace(ch, '')

    # Strip non-Hebrew/non-Latin characters
    script = re.sub(r'[^\u0020-\u007F\u05D0-\u05FF\u05B0-\u05C7\u200F\n\r.,!?;:()\-–—\'\"״׳ ]', ' ', script)
    script = re.sub(r'  +', ' ', script)

    # ── Validation ──
    word_count = len(script.split())
    if word_count < 100:
        print(f"  ⚠️  Script too short ({word_count} words), using fallback")
        return _generate_podcast_fallback(data)
    if not any(c in script for c in 'אבגדהוזחטיכלמנסעפצקרשת'):
        print(f"  ⚠️  Script not in Hebrew, using fallback")
        return _generate_podcast_fallback(data)

    print(f"  ✅ Podcast script: full LLM, {len(valid_stocks)} stocks, {len(script)} chars, {word_count} words")
    return script


def _generate_podcast_fallback(data):
    """Generate basic Hebrew podcast script without AI."""
    m = data['macro']
    stocks = data['stocks']
    valid_stocks = [s for s in stocks if 'error' not in s]
    date = datetime.fromisoformat(data['date']).strftime('%d/%m/%Y')

    fg = m.get('fear_greed', {})
    sp = m.get('sp500', {})

    script = f"""שלום וברוכים הבאים לסיכום היומי של שוק ההון, תאריך {date}. תזכורת — זו אינה המלצת השקעה, אלא סיכום חדשות ומידע בלבד.

"""

    # Macro — tell the story, not the numbers
    script += "נתחיל בתמונה הרחבה. "

    if sp.get('price'):
        sp_change = sp.get('change_pct', 0)
        if sp_change < -1.5:
            script += f"יום אדום בוול סטריט — מדד ה-S&P 500 ירד {abs(sp_change):.1f} אחוז. "
        elif sp_change > 1.5:
            script += f"יום ירוק בוול סטריט — מדד ה-S&P 500 עלה {sp_change:.1f} אחוז. "
        else:
            script += f"השווקים בוול סטריט נסחרו ללא כיוון ברור, שינוי של {sp_change:+.1f} אחוז ב-S&P 500. "

    if fg.get('score') is not None:
        score = fg['score']
        if score < 20:
            script += "הסנטימנט בשוק הוא פחד קיצוני — משקיעים חוששים ומוכרים. היסטורית, דווקא ברגעים כאלה נוצרות הזדמנויות למי שמוכן לחכות. "
        elif score > 75:
            script += "הסנטימנט בשוק חיובי מאוד, אולי אפילו יותר מדי — תאוות בצע שולטת, וזה בדרך כלל הזמן להיזהר. "
        elif score < 35:
            script += "יש חששות בשוק, הסנטימנט נמוך יחסית. "

    if m.get('vix') and m['vix'] > 25:
        script += "התנודתיות גבוהה — מה שאומר שהשוק עצבני ויש חוסר ודאות. "

    if m.get('usd_ils'):
        script += f"שער הדולר עומד על {m['usd_ils']:.2f} שקלים. "

    # Stocks — winners and losers as a story
    winners = [s for s in valid_stocks if s.get('change_pct', 0) > 1]
    losers = [s for s in valid_stocks if s.get('change_pct', 0) < -1]

    script += "\n\nעכשיו למניות בתיק — מה קרה היום.\n"

    if losers:
        losers.sort(key=lambda s: s['change_pct'])
        script += "\nמהצד המאכזב: "
        for s in losers[:3]:
            script += f"{s['hebrew']} ירדה {abs(s['change_pct']):.1f} אחוז. "

    if winners:
        winners.sort(key=lambda s: s['change_pct'], reverse=True)
        script += "\nמהצד המעודד: "
        for s in winners[:3]:
            script += f"{s['hebrew']} עלתה {s['change_pct']:.1f} אחוז. "

    # Per-stock — focus on news and story
    script += "\n\nבואו נצלול פנימה:\n"

    for s in valid_stocks[:8]:
        script += f"\n{s['hebrew']} — "
        change = s.get('change_pct', 0)
        if abs(change) > 0.5:
            direction = "עלתה" if change > 0 else "ירדה"
            script += f"{direction} {abs(change):.1f} אחוז ונסחרת סביב {s['currency']}{s['price']}. "
        else:
            script += f"נסחרת סביב {s['currency']}{s['price']}, ללא שינוי משמעותי. "

        # Include news summary if available
        news_summary = s.get('news_summary', {})
        if news_summary.get('has_news') and news_summary.get('summary'):
            script += news_summary['summary'] + " "
        else:
            # Fallback to basic analysis without news
            if s.get('target_mean'):
                upside = round((s['target_mean'] - s['price']) / s['price'] * 100, 1)
                if upside > 15:
                    script += f"האנליסטים עדיין אופטימיים — יעד ממוצע של {s['currency']}{s['target_mean']}, שזה פוטנציאל של {upside:.0f} אחוז. "
                elif upside < -5:
                    script += f"האנליסטים חושבים שהמחיר גבוה — היעד הממוצע נמוך מהמחיר הנוכחי. "

            if s.get('revenue_growth') and abs(s['revenue_growth']) > 5:
                if s['revenue_growth'] > 15:
                    script += f"מבחינת צמיחה — ההכנסות צומחות ב-{s['revenue_growth']:.0f} אחוז, מה שמעיד על ביקוש חזק. "
                elif s['revenue_growth'] < -5:
                    script += f"ההכנסות ירדו ב-{abs(s['revenue_growth']):.0f} אחוז, מה שמדאיג. "

    # Summary — mood, not numbers
    script += "\n\nלסיכום: "

    if (fg.get('score') or 50) < 25:
        script += "השוק בגדול חושש. אם אתם משקיעים לטווח ארוך, זה דווקא יכול להיות הזמן להיות אמיצים, אבל בזהירות. "
    elif (fg.get('score') or 50) > 70:
        script += "השוק בסנטימנט חיובי, אבל כדאי לזכור שבשיא האופטימיות צריך דווקא להיזהר. "
    else:
        script += "השוק ממשיך להתנהל, חשוב לעקוב אחרי החדשות ולהישאר מעודכנים. "

    script += "\nזה הסיכום להיום. יום מוצלח ותודה שהאזנתם!"

    return script


def _number_to_hebrew(n):
    """Convert an integer (0-999999) to Hebrew words."""
    if n == 0:
        return 'אפס'
    _ones = ['', 'אחד', 'שניים', 'שלושה', 'ארבעה', 'חמישה', 'שישה', 'שבעה', 'שמונה', 'תשעה']
    _teens = ['עשרה', 'אחד עשר', 'שניים עשר', 'שלושה עשר', 'ארבעה עשר', 'חמישה עשר',
              'שישה עשר', 'שבעה עשר', 'שמונה עשר', 'תשעה עשר']
    _tens = ['', 'עשר', 'עשרים', 'שלושים', 'ארבעים', 'חמישים', 'שישים', 'שבעים', 'שמונים', 'תשעים']

    parts = []
    if n >= 1000:
        thousands = n // 1000
        n %= 1000
        if thousands == 1:
            parts.append('אלף')
        elif thousands == 2:
            parts.append('אלפיים')
        elif thousands <= 9:
            parts.append(f'{_ones[thousands]} אלפים')
        else:
            parts.append(f'{thousands} אלף')

    if n >= 100:
        hundreds = n // 100
        n %= 100
        if hundreds == 1:
            parts.append('מאה')
        elif hundreds == 2:
            parts.append('מאתיים')
        else:
            parts.append(f'{_ones[hundreds]} מאות')

    if n >= 10 and n < 20:
        parts.append(_teens[n - 10])
    else:
        if n >= 20:
            parts.append(_tens[n // 10])
            n %= 10
        if n > 0:
            parts.append(_ones[n])

    return ' '.join(parts)


def _prepare_text_for_tts(text):
    """Prepare text for better Hebrew TTS pronunciation."""
    import re

    # Stock ticker pronunciation map — how to say them in Hebrew
    # Longer matches first to avoid partial replacements (e.g. GOOGL before GOOG)
    ticker_pronunciation = [
        # Common portfolio tickers
        ('AAPL', 'אפל'),
        ('TSLA', 'טסלה'),
        ('NVDA', 'אנבידיה'),
        ('GOOGL', 'גוגל'),
        ('GOOG', 'גוגל'),
        ('MSFT', 'מייקרוסופט'),
        ('AMZN', 'אמזון'),
        ('META', 'מטא'),
        ('NFLX', 'נטפליקס'),
        ('ORCL', 'אורקל'),
        ('CRM', 'סיילספורס'),
        ('ADBE', 'אדובי'),
        ('INTC', 'אינטל'),
        ('AMD', 'איי אם די'),
        ('QCOM', 'קוואלקום'),
        ('AVGO', 'ברודקום'),
        ('JPM', 'ג\'יי פי מורגן'),
        ('BAC', 'בנק אוף אמריקה'),
        ('GS', 'גולדמן זאקס'),
        ('MS', 'מורגן סטנלי'),
        ('WFC', 'וולס פארגו'),
        ('JNJ', 'ג\'ונסון אנד ג\'ונסון'),
        ('PFE', 'פייזר'),
        ('UNH', 'יונייטד הלת'),
        ('LLY', 'אלי לילי'),
        ('XOM', 'אקסון מוביל'),
        ('CVX', 'שברון'),
        ('SOFI', 'סופי'),
        ('PANW', 'פאלו אלטו'),
        ('ONON', 'און'),
        ('IREN', 'אירן'),
        ('MELI', 'מרקדו ליברה'),
        ('IBM', 'איי בי אם'),
        # Indices and terms
        ('S&P 500', 'אס אנד פי חמש מאות'),
        ('S&P', 'אס אנד פי'),
        ('VIX', 'ויקס'),
        ('RSI', 'מדד חוזק יחסי'),
        ('CNBC', 'סי אן בי סי'),
        ('GPU', 'ג\'י פי יו'),
        ('AI', 'בינה מלאכותית'),
        ('CEO', 'מנכ"ל'),
        ('CFO', 'סמנכ"ל כספים'),
        ('EPS', 'רווח למניה'),
        ('CPI', 'מדד המחירים לצרכן'),
        ('USD', 'דולר'),
        ('ILS', 'שקל'),
        ('IPO', 'הנפקה'),
        ('ETF', 'תעודת סל'),
        ('MA200', 'ממוצע מאתיים יום'),
        ('MA50', 'ממוצע חמישים יום'),
    ]

    # ── Step 1: Strip non-Hebrew/non-Latin/non-punctuation characters (Chinese, Arabic etc.) ──
    # Keep: Hebrew (0590-05FF), Latin (0020-007F), common punctuation, digits
    text = re.sub(r'[^\u0020-\u007F\u05D0-\u05FF\u05B0-\u05C7\u200F\n\r.,!?;:()\-–—\'\"״׳ ]', ' ', text)
    text = re.sub(r'  +', ' ', text)  # collapse multiple spaces

    # ── Step 2: Fix broken spacing around dashes before terms ──
    # "ב- AI" → "ב-AI", "ל- AI" etc.
    text = re.sub(r'([בלכמ])-\s+', r'\1-', text)

    # ── Step 3: Replace English tickers/terms with Hebrew pronunciation (longer first) ──
    # Must come before generic English cleanup so specific terms are caught
    extra_terms = [
        ('OpenAI', 'אופן-איי'),
        ('Azure', "אז'ור"),
        ('ChatGPT', "צ'ט-ג'י-פי-טי"),
        ('chatGPT', "צ'ט-ג'י-פי-טי"),
        ('chatgpt', "צ'ט-ג'י-פי-טי"),
        ('Copilot', "קו-פיילוט"),
        ('Stargate', 'סטארגייט'),
        ('xAI', 'אקס-איי'),
        ('Grok', "גרוק"),
        ('Gemini', "ג'מיני"),
        ('Nvidia', 'אנבידיה'),
        ('nvidia', 'אנבידיה'),
        # Financial terms the LLM might use in English
        ('Jefferies', "ג'פריז"),
        ('Barclays', 'ברקליז'),
        ('Goldman Sachs', 'גולדמן זאקס'),
        ('Wells Fargo', 'וולס פארגו'),
        ('William Blair', 'ויליאם בלייר'),
        ('UBS', 'יו בי אס'),
        ('JP Morgan', "ג'יי פי מורגן"),
    ]
    for eng, heb in extra_terms:
        text = text.replace(eng, heb)

    for eng, heb in ticker_pronunciation:
        text = text.replace(eng, heb)

    # ── Step 4: Replace remaining ALL-CAPS English words (3+ letters) with spaced letters ──
    def _spell_out(m):
        word = m.group(0)
        return ' '.join(word)  # "EMJ" → "E M J" — TTS reads each letter separately
    text = re.sub(r'\b[A-Z]{2,6}\b', _spell_out, text)

    # ── Step 5: Numbers to Hebrew words (BEFORE symbol replacements) ──
    # This is the single biggest TTS pronunciation improvement.

    # Dollar amounts: $1,234.56 → "אלף מאתיים שלושים וארבעה דולר וחמישים ושישה סנט"
    def _dollar_to_hebrew(m):
        raw = m.group(1).replace(',', '')
        if '.' in raw:
            whole, frac = raw.split('.', 1)
        else:
            whole, frac = raw, ''
        whole_int = int(whole) if whole else 0
        heb = _number_to_hebrew(whole_int)
        if frac and int(frac) > 0:
            return f'{heb} דולר ו{_number_to_hebrew(int(frac))} סנט'
        return f'{heb} דולר'
    text = re.sub(r'\$(\d[\d,]*\.?\d*)', _dollar_to_hebrew, text)

    # Shekel amounts: ₪72.4 → "שבעים ושניים שקל וארבעים אגורות"
    def _shekel_to_hebrew(m):
        raw = m.group(1).replace(',', '')
        if '.' in raw:
            whole, frac = raw.split('.', 1)
        else:
            whole, frac = raw, ''
        whole_int = int(whole) if whole else 0
        heb = _number_to_hebrew(whole_int)
        if frac and int(frac) > 0:
            return f'{heb} שקל'
        return f'{heb} שקל'
    text = re.sub(r'₪(\d[\d,]*\.?\d*)', _shekel_to_hebrew, text)

    # Percentages: +2.6% → "פלוס שתיים נקודה שש אחוז"
    def _pct_to_hebrew(m):
        sign = m.group(1) or ''
        num = m.group(2)
        prefix = 'פלוס ' if sign == '+' else ('מינוס ' if sign == '-' else '')
        if '.' in num:
            whole, frac = num.split('.', 1)
            whole_heb = _number_to_hebrew(int(whole)) if whole else 'אפס'
            frac_heb = _number_to_hebrew(int(frac)) if frac else ''
            return f'{prefix}{whole_heb} נקודה {frac_heb} אחוז'
        return f'{prefix}{_number_to_hebrew(int(num))} אחוז'
    text = re.sub(r'([+-])?(\d+\.?\d*)%', _pct_to_hebrew, text)

    # Standalone numbers (e.g. years "2026", counts "35")
    def _num_to_hebrew(m):
        raw = m.group(0).replace(',', '')
        if '.' in raw:
            whole, frac = raw.split('.', 1)
            whole_heb = _number_to_hebrew(int(whole))
            frac_heb = _number_to_hebrew(int(frac)) if frac else ''
            return f'{whole_heb} נקודה {frac_heb}'
        n = int(raw)
        # Years: keep as-is for TTS (2026 is fine)
        if 1900 < n < 2100:
            return raw
        if n > 999999:
            return raw  # too large, let TTS handle it
        return _number_to_hebrew(n)
    text = re.sub(r'\b\d{1,6}(?:,\d{3})*(?:\.\d+)?\b', _num_to_hebrew, text)

    # ── Step 6: Punctuation / flow fixes for TTS ──
    # Em-dash and en-dash → comma (avoids long unnatural pauses)
    text = re.sub(r'\s*[—–]\s*', ', ', text)
    # Ellipsis → period
    text = text.replace('...', '. ')
    # Markdown artifacts
    text = text.replace('**', '').replace('*', '').replace('#', '')
    text = text.replace('[', '').replace(']', '')
    # Double periods
    text = re.sub(r'\.{2,}', '.', text)
    # Multiple spaces/commas
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'  +', ' ', text)

    return text.strip()


def _add_niqqud(text):
    """Add niqqud (vowel diacritics) to Hebrew text using LLM."""
    try:
        print("  📝 Adding niqqud...")
        # Split into chunks if text is very long (LLM token limit)
        max_chunk = 1500
        if len(text) <= max_chunk:
            chunks = [text]
        else:
            # Split on paragraph boundaries
            paragraphs = text.split('\n\n')
            chunks = []
            current = ""
            for p in paragraphs:
                if len(current) + len(p) > max_chunk and current:
                    chunks.append(current.strip())
                    current = p
                else:
                    current += ("\n\n" if current else "") + p
            if current.strip():
                chunks.append(current.strip())

        result_parts = []
        for chunk in chunks:
            nikud = _llm_chat(
                [{"role": "user", "content": chunk}],
                max_tokens=4000,
                system="הוסף ניקוד מלא לטקסט העברי. החזר רק את הטקסט עם ניקוד, בלי שום הסבר, בלי שינויים בתוכן. שמור על כל סימני הפיסוק, מספרים ומילים לועזיות כפי שהם."
            )
            result_parts.append(nikud.strip())

        result = '\n\n'.join(result_parts)
        print(f"  ✅ Niqqud added ({len(result)} chars)")
        return result
    except Exception as e:
        print(f"  ⚠️  Niqqud failed ({e}), using text without niqqud")
        return text


_ELEVENLABS_CHUNK_LIMIT = 2400  # eleven_v3 max chars per call (safe margin under 2500)


def _elevenlabs_chunk(api_key, voice_id, text):
    """Send one chunk to ElevenLabs and return raw MP3 bytes, or raise on error."""
    resp = requests.post(
        f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
        headers={
            'xi-api-key': api_key,
            'Content-Type': 'application/json',
            'Accept': 'audio/mpeg',
        },
        json={
            'text': text,
            'model_id': 'eleven_v3',
            'voice_settings': {
                'stability': 0.35,
                'similarity_boost': 0.80,
                'style': 0.50,
                'use_speaker_boost': True,
            },
            'language_code': 'he',
        },
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs {resp.status_code}: {resp.text[:200]}")
    return resp.content


def _elevenlabs_tts(text, output_path):
    """ElevenLabs TTS — splits long scripts into paragraph chunks to avoid cutoff.

    Model: eleven_v3 (only ElevenLabs model with Hebrew support, 74 languages).
    Splits on paragraph boundaries when text > _ELEVENLABS_CHUNK_LIMIT chars.
    Concatenates MP3 bytes directly (works with all players).
    """
    api_key = os.environ.get('ELEVENLABS_API_KEY', '')
    if not api_key:
        return False

    voice_id = os.environ.get('ELEVENLABS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL')  # Sarah default

    try:
        # Split into paragraph chunks if needed
        if len(text) <= _ELEVENLABS_CHUNK_LIMIT:
            chunks = [text]
        else:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            chunks = []
            current = ''
            for para in paragraphs:
                # If single paragraph exceeds limit, split on sentence boundaries
                if len(para) > _ELEVENLABS_CHUNK_LIMIT:
                    sentences = re.split(r'(?<=[.!?,])\s+', para)
                    for sent in sentences:
                        if len(current) + len(sent) + 1 > _ELEVENLABS_CHUNK_LIMIT and current:
                            chunks.append(current.strip())
                            current = sent
                        else:
                            current = (current + ' ' + sent).strip()
                elif len(current) + len(para) + 2 > _ELEVENLABS_CHUNK_LIMIT and current:
                    chunks.append(current.strip())
                    current = para
                else:
                    current = (current + '\n\n' + para).strip() if current else para
            if current.strip():
                chunks.append(current.strip())

        print(f"  🎙️  ElevenLabs: {len(chunks)} chunk(s), {len(text)} chars total")

        # Generate audio for each chunk and concatenate bytes
        all_bytes = b''
        for i, chunk in enumerate(chunks):
            print(f"    chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            all_bytes += _elevenlabs_chunk(api_key, voice_id, chunk)

        with open(str(output_path), 'wb') as f:
            f.write(all_bytes)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  🎙️  Audio saved (ElevenLabs eleven_v3 / {voice_id}): {Path(output_path).name} ({size_kb:.0f} KB)")
        return True

    except Exception as e:
        print(f"  ⚠️  ElevenLabs TTS failed ({e})")
        return False


def text_to_speech(text, output_path, voice=None):
    """Convert text to speech — ElevenLabs (if key set) → Edge TTS → gTTS.

    Available Hebrew Edge TTS voices:
      he-IL-AvriNeural  — male, clear and professional
      he-IL-HilaNeural  — female, warm and natural
    """
    # Step 1: Preprocess (replace tickers, $, % etc)
    text = _prepare_text_for_tts(text)

    # Step 2: ElevenLabs — only if explicitly enabled (costs money)
    if os.environ.get('ELEVENLABS_API_KEY') and os.environ.get('ELEVENLABS_ENABLED', '').lower() == 'true':
        if _elevenlabs_tts(text, output_path):
            return True
        print("  ↩️  Falling back to Edge TTS")

    # Step 3: Edge TTS (free, Microsoft neural voices) — split into paragraphs to prevent repetition bug
    try:
        import asyncio
        import tempfile
        import edge_tts

        _voice = voice or os.environ.get('PODCAST_VOICE', 'he-IL-HilaNeural')

        # Split on paragraph breaks to avoid Edge TTS internal chunking (causes repeats)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        async def _generate_paragraph(para, out_file):
            communicate = edge_tts.Communicate(para, _voice, rate="-10%")
            await communicate.save(out_file)

        def _run_async(coro):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(lambda: asyncio.run(coro)).result(timeout=60)
                else:
                    return loop.run_until_complete(coro)
            except RuntimeError:
                return asyncio.run(coro)

        if len(paragraphs) == 1:
            # Single chunk — no concat needed
            _run_async(_generate_paragraph(paragraphs[0], str(output_path)))
        else:
            # Generate each paragraph → temp file → ffmpeg concat
            tmp_files = []
            try:
                for i, para in enumerate(paragraphs):
                    tf = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                    tf.close()
                    tmp_files.append(tf.name)
                    _run_async(_generate_paragraph(para, tf.name))

                # Concatenate MP3 chunks — try ffmpeg first, fall back to byte concat
                import subprocess
                try:
                    list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                    for tf_name in tmp_files:
                        list_file.write(f"file '{tf_name}'\n")
                    list_file.close()
                    result = subprocess.run(
                        ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                         '-i', list_file.name, '-c', 'copy', str(output_path)],
                        capture_output=True, timeout=120
                    )
                    os.unlink(list_file.name)
                    if result.returncode != 0:
                        raise RuntimeError("ffmpeg failed")
                except (FileNotFoundError, RuntimeError):
                    # ffmpeg not available — concatenate raw bytes (works for most players)
                    with open(str(output_path), 'wb') as out_f:
                        for tf_name in tmp_files:
                            with open(tf_name, 'rb') as in_f:
                                out_f.write(in_f.read())
            finally:
                for tf_name in tmp_files:
                    try:
                        os.unlink(tf_name)
                    except OSError:
                        pass

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  🔊 Audio saved (Edge TTS {_voice}): {Path(output_path).name} ({size_kb:.0f} KB)")
        return True
    except ImportError:
        print("  ⚠️  edge-tts not installed, falling back to gTTS")
    except Exception as e:
        print(f"  ⚠️  Edge TTS failed ({e}), falling back to gTTS")

    # Fallback to gTTS (without SSML)
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='iw', slow=False)
        tts.save(str(output_path))
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  🔊 Audio saved (gTTS): {output_path.name} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  ❌ TTS failed: {e}")
        return False


# ─── Main ───

def run(portfolio=None):
    """Run the daily briefing."""
    if portfolio:
        global PORTFOLIO  # noqa: PLW0603
        PORTFOLIO = portfolio

    today = datetime.now().strftime('%Y-%m-%d')

    # Collect data
    data = collect_all()

    # Save raw data
    data_path = OUTPUT_DIR / f'briefing_{today}_data.json'
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 Data: {data_path}")

    # Generate text report
    report = generate_text_report(data)
    report_path = OUTPUT_DIR / f'briefing_{today}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"📄 Report: {report_path}")

    # Print to console
    print("\n" + report)

    # Generate podcast
    script = generate_podcast_script(data)
    script_path = OUTPUT_DIR / f'briefing_{today}_script.txt'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    print(f"\n📝 Script: {script_path}")

    # Convert to audio
    audio_path = OUTPUT_DIR / f'briefing_{today}.mp3'
    text_to_speech(script, audio_path)

    print(f"\n{'='*60}")
    print(f"✅ Daily briefing complete!")
    print(f"   📄 {report_path}")
    print(f"   🎵 {audio_path}")
    print(f"{'='*60}")

    return {
        'report_path': str(report_path),
        'audio_path': str(audio_path),
        'data_path': str(data_path),
    }


if __name__ == '__main__':
    # --collect-json OUTPUT_PATH: run collect_all() and write JSON to a file (used by Flask subprocess)
    if '--collect-json' in sys.argv:
        _idx = sys.argv.index('--collect-json')
        _out_path = sys.argv[_idx + 1] if _idx + 1 < len(sys.argv) else None
        tickers_env = os.environ.get('_BRIEFING_TICKERS', '')
        if tickers_env:
            PORTFOLIO = [t.strip() for t in tickers_env.split(',') if t.strip()]  # noqa: PLW0603
        import json as _json
        _data = collect_all()
        if _out_path:
            with open(_out_path, 'w', encoding='utf-8') as _f:
                _json.dump(_data, _f, ensure_ascii=False, default=str)
        else:
            sys.stdout.write(_json.dumps(_data, ensure_ascii=False, default=str))
            sys.stdout.flush()
        sys.exit(0)
    # --report-only: collect data + generate text report only (no script/TTS)
    _report_only = '--report-only' in sys.argv
    _args = [a for a in sys.argv[1:] if a != '--report-only']
    # Allow custom tickers from command line
    if _args:
        tickers = [t.strip() for t in ' '.join(_args).replace(',', ' ').split()]
    else:
        tickers = None
    if _report_only:
        if tickers:
            PORTFOLIO = tickers
        today = datetime.now().strftime('%Y-%m-%d')
        data = collect_all()
        data_path = OUTPUT_DIR / f'briefing_{today}_data.json'
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Data: {data_path}")
        report = generate_text_report(data)
        report_path = OUTPUT_DIR / f'briefing_{today}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Report: {report_path}")
        print(report)
    else:
        run(tickers)
