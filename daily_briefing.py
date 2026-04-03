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
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import yfinance as yf
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

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
        macro['ta35'] = round(safe_get(ti, 'regularMarketPrice', 'currentPrice') or 0, 2)
    except:
        macro['ta35'] = None

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

    # RSI (Wilder's smoothing) + MA200
    hist_long = t.history(period='2y', interval='1d')
    rsi = None
    ma200 = None
    if not hist_long.empty and len(hist_long) > 30:
        rsi = _wilder_rsi(hist_long['Close'].values, 14)
    if not hist_long.empty and len(hist_long) >= 200:
        ma200_val = hist_long['Close'].rolling(200).mean().iloc[-1]
        if not np.isnan(ma200_val):
            ma200 = round(float(ma200_val), 2)

    # 52-week range
    w52_high = safe_get(info, 'fiftyTwoWeekHigh')
    w52_low = safe_get(info, 'fiftyTwoWeekLow')

    # Distance from 52w high
    pct_from_high = round((price - w52_high) / w52_high * 100, 1) if w52_high and price else None

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


def _tavily_extract_urls(tavily_client, urls):
    """Use Tavily extract to get full article content from URLs."""
    try:
        # Tavily extract API — pulls full text from URLs
        result = tavily_client.extract(urls=urls[:5])
        extracted = {}
        for r in result.get('results', []):
            url = r.get('url', '')
            text = r.get('raw_content', '') or r.get('text', '')
            if url and text and len(text) > 100:
                extracted[url] = text[:3000]  # Cap per article
        return extracted
    except Exception as e:
        print(f"    ⚠️  Tavily extract failed: {e}")
        return {}


def collect_news_deep(ticker, resolved_ticker, company_name, hebrew_name):
    """Collect news from 3 sources with actual article content."""
    articles = []
    seen_urls = set()
    rss_urls_to_extract = []  # URLs from RSS that need full content

    # Source 1: Yahoo Finance RSS
    if feedparser:
        try:
            feed = feedparser.parse(f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={resolved_ticker}')
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
            feed = feedparser.parse(gn_url)
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

    # Use Tavily extract to get full content for RSS articles
    tavily_key = os.environ.get('TAVILY_API_KEY', '')
    if tavily_key and TavilyClient and rss_urls_to_extract:
        tavily = TavilyClient(api_key=tavily_key)
        extracted = _tavily_extract_urls(tavily, rss_urls_to_extract[:5])
        for a in articles:
            url = a.get('url', '')
            if url in extracted:
                a['content'] = extracted[url]
        time.sleep(0.3)

    # Source 3: Tavily Search (deep content extraction)
    if tavily_key and TavilyClient:
        try:
            tavily = TavilyClient(api_key=tavily_key)
            query = f"{company_name} ({ticker}) stock news latest developments"
            results = tavily.search(query=query, search_depth="advanced", max_results=5,
                                     include_raw_content=True)
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
            time.sleep(0.5)
        except Exception as e:
            print(f"    ⚠️  Tavily failed for {ticker}: {e}")

    # For Israeli stocks, also search in Hebrew
    is_israeli = resolved_ticker.endswith('.TA')
    if is_israeli and tavily_key and TavilyClient:
        try:
            tavily = TavilyClient(api_key=tavily_key)
            query = f"{hebrew_name} מניה חדשות"
            results = tavily.search(query=query, search_depth="advanced", max_results=3,
                                     include_raw_content=True)
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
            time.sleep(0.5)
        except Exception as e:
            print(f"    ⚠️  Tavily HE failed for {ticker}: {e}")

    return articles


DEEPSEEK_MODEL = "deepseek-chat"
GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"


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
    """Call LLM — tries DeepSeek first, then Gemini, then Groq."""
    deepseek_key = os.environ.get('DEEPSEEK_API_KEY', '')
    gemini_key = os.environ.get('GEMINI_API_KEY', '')
    groq_key = os.environ.get('GROQ_API_KEY', '')

    # 1. DeepSeek (primary)
    if deepseek_key:
        try:
            return _deepseek_chat(deepseek_key, messages, max_tokens, system)
        except Exception as e:
            print(f"    ⚠️  DeepSeek failed: {e}")

    # 2. Gemini (fallback)
    if gemini_key:
        try:
            print(f"    🔄 Trying Gemini...")
            return _gemini_chat(gemini_key, messages, max_tokens, system)
        except Exception as e:
            print(f"    ⚠️  Gemini failed: {e}")

    # 3. Groq (last resort)
    if groq_key:
        print(f"    🔄 Trying Groq...")
        return _groq_chat(groq_key, messages, max_tokens, system)

    raise RuntimeError("No LLM API key configured (set DEEPSEEK_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY)")


def summarize_stock_news(ticker, hebrew_name, stock_data, articles):
    """Summarize news articles for a single stock using LLM."""
    # Only summarize if we have articles with actual content
    articles_with_content = [a for a in articles if a.get('content') and len(a['content']) > 50]
    if not articles_with_content:
        return {
            'summary': '',
            'has_news': False,
            'article_count': len(articles),
            'key_headlines': [a['title'] for a in articles[:3]],
        }

    try:
        change_pct = stock_data.get('change_pct', 0)

        articles_text = ""
        for i, a in enumerate(articles_with_content[:5], 1):
            content = a['content'][:2000]
            articles_text += f"\n--- כתבה {i}: {a['title']} ({a['source']}) ---\n{content}\n"

        prompt = f"""אתה מגיש פודקאסט כלכלי יומי בעברית. הסגנון שלך כמו מגיש פודקאסט ישראלי טוב — מקצועי, ברור, מדבר בגובה העיניים. לא קריין חדשות רשמי, אבל גם לא חבר'ה מהשכונה.

המניה: {hebrew_name} ({ticker})
מחיר: {stock_data.get('currency', '$')}{stock_data.get('price', '?')} | שינוי: {change_pct:+.1f}%

כתבות:
{articles_text}

כתוב 200-350 מילים. זה פודקאסט — טקסט שמיועד להקראה בקול.

הנחיות:
- ספר את הסיפור של המניה היום כנרטיב אחד זורם. אסור "הכתבה הראשונה", "כתבה נוספת", "לפי כתבה ש..."
- שזור את כל המידע מהכתבות ביחד — מה קרה, למה, ומה המשמעות
- ציין פרטים ספציפיים: שמות אנשים, מספרים, סכומים, אירועים
- אם מישהו מכר מניות — ציין מי, כמה, ולמה. אם אנליסט אמר משהו — ציין שמו ומה אמר
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

    # Phase 2: Deep news collection + summarization
    print("\n📰 Collecting deep news...")
    has_llm = os.environ.get('DEEPSEEK_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GROQ_API_KEY', '')
    for s in stocks:
        if 'error' in s:
            continue
        print(f"  📰 {s['ticker']}...", end=" ", flush=True)
        articles = collect_news_deep(s['ticker'], s['resolved'], s.get('name', ''), s.get('hebrew', ''))
        s['news_deep'] = articles
        content_count = len([a for a in articles if a.get('content') and len(a['content']) > 50])
        print(f"{len(articles)} articles ({content_count} with content)")

        if has_llm and content_count >= 1:
            s['news_summary'] = summarize_stock_news(s['ticker'], s['hebrew'], s, articles)
            if s['news_summary']['has_news']:
                print(f"    ✅ Summary generated")
        else:
            s['news_summary'] = {'summary': '', 'has_news': False, 'article_count': len(articles), 'key_headlines': []}

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


def generate_text_report(data):
    """Generate Hebrew text report."""
    m = data['macro']
    stocks = data['stocks']
    date = datetime.fromisoformat(data['date']).strftime('%d/%m/%Y')

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"📊 סיכום יומי לתיק ההשקעות — {date}")
    lines.append(f"{'='*60}")
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
        lines.append(f"  🇮🇱 TA-35: {m['ta35']:,.2f}")

    lines.append("")

    # ─── Summary signal ───
    fg_score = fg.get('score') or 50
    vix_val = m.get('vix') or 20
    sp_rsi = sp.get('rsi') or 50

    if fg_score < 20 and vix_val > 30:
        lines.append("⚠️  אזהרה: פחד קיצוני בשוק — VIX גבוה + Fear & Greed נמוך מאוד")
        lines.append("   יכול להיות הזדמנות קנייה למשקיעים לטווח ארוך")
    elif fg_score > 75:
        lines.append("⚠️  זהירות: תאוות בצע בשוק — שקול לקחת רווחים")

    lines.append("")

    # ─── Stocks ───
    lines.append("📈 ניתוח מניות בתיק")
    lines.append("─" * 40)

    # Sort by change
    valid_stocks = [s for s in stocks if 'error' not in s]
    valid_stocks.sort(key=lambda s: s.get('change_pct', 0))

    # Winners & Losers
    winners = [s for s in valid_stocks if s.get('change_pct', 0) > 0]
    losers = [s for s in valid_stocks if s.get('change_pct', 0) <= 0]

    if losers:
        lines.append("")
        lines.append("  🔴 ירידות:")
        for s in losers:
            lines.append(f"    {s['hebrew']} ({s['ticker']}): {s['currency']}{s['price']} ({s['change_pct']:+.1f}%)")

    if winners:
        lines.append("")
        lines.append("  🟢 עליות:")
        for s in reversed(winners):
            lines.append(f"    {s['hebrew']} ({s['ticker']}): {s['currency']}{s['price']} ({s['change_pct']:+.1f}%)")

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


BRIEFING_SYSTEM_PROMPT = """אתה מגיש פודקאסט כלכלי יומי בעברית. הסגנון שלך כמו מגיש פודקאסט ישראלי מקצועי — ברור, נגיש, מעניין. לא קריין חדשות רשמי, אבל גם לא סלנג ולא "אחי". מגיש שמכבד את המאזין ומדבר איתו בגובה העיניים.

הסגנון שלך:
- אתה מספר את הסיפור של כל מניה — מה קרה, למה, ומה זה אומר
- אם מנכ"ל אמר משהו — ספר מה. אם מישהו מכר — ספר מי ולמה. פרטים ספציפיים עושים את ההבדל
- תיצור עניין דרך התוכן עצמו, לא דרך ביטויים מאולצים
- שפה ברורה ופשוטה, בלי מונחים טכניים, בלי סלנג
- תרגם הכל לעברית, אסור אנגלית

כללים טכניים:
- אסור: סוגריים מרובעים, כוכביות, סולמית, מספור, markdown
- אורך מינימלי: 1200 מילים
- טקסט רציף מוכן להקראה בקול רם
- אסור: "הכתבה הראשונה", "כתבה נוספת", "לפי כתבה ש..." — שזור את המידע בנרטיב טבעי
- אל תחזור על אותו רעיון פעמיים"""


def generate_podcast_script(data):
    """Generate Hebrew podcast script — tries LLM API first, falls back to basic."""
    has_llm = os.environ.get('DEEPSEEK_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GROQ_API_KEY', '')

    if has_llm:
        try:
            return _generate_podcast_with_llm(data)
        except Exception as e:
            print(f"  ⚠️  LLM API failed for briefing: {e}, using fallback")

    return _generate_podcast_fallback(data)


def _generate_podcast_with_llm(data):
    """Hybrid podcast: structured skeleton + LLM for intro/transitions/closing."""

    m = data['macro']
    stocks = data['stocks']
    valid_stocks = [s for s in stocks if 'error' not in s]
    date = datetime.fromisoformat(data['date']).strftime('%d/%m/%Y')

    # Sort: biggest movers first
    valid_stocks.sort(key=lambda s: abs(s.get('change_pct', 0)), reverse=True)

    # --- Step 1: Use Groq to generate intro + closing based on the news ---
    # Build a brief summary of what happened for the intro
    top_stories = []
    for s in valid_stocks[:5]:
        ns = s.get('news_summary', {})
        if ns.get('has_news') and ns.get('summary'):
            summary_first_line = ns['summary'].split('\n')[0][:150]
            top_stories.append(f"{s.get('hebrew', s['ticker'])}: {summary_first_line}")

    intro_prompt = f"""כתוב פתיחה לפודקאסט כלכלי יומי בעברית (80-120 מילים).

תאריך: {date}
S&P 500: {m.get('sp500', {}).get('price', '?')} ({m.get('sp500', {}).get('change_pct', 0):+.1f}%)
VIX: {m.get('vix', '?')}
דולר-שקל: {m.get('usd_ils', '?')}

הסיפורים המרכזיים היום:
{chr(10).join(top_stories)}

מבנה חובה:
1. פתח עם ברכה קצרה וטבעית + התאריך
2. מיד אחר כך חובה לומר: "תזכורת חשובה — מה שאני אומר כאן זה לא ייעוץ השקעות, אלא סיכום חדשות ומידע בלבד."
3. תאר את מצב הרוח בשוק בשני-שלושה משפטים — כסיפור, לא כרשימת מספרים
4. תן טיזר קצר — מה הסיפור המעניין ביותר שנדבר עליו היום

טון: מגיש פודקאסט מקצועי ונגיש. לא רשמי מדי, לא קז'ואל מדי.
אסור: סלנג, "אחי", "שמע". אסור: כוכביות/מספור/markdown/אנגלית."""

    intro = _llm_chat([{"role": "user", "content": intro_prompt}], max_tokens=500)

    # --- Step 2: Build the stock sections from the rich news summaries ---
    stock_sections = []
    transitions = [
        "נעבור ל",
        "ועכשיו ל",
        "נמשיך ל",
        "ומה קורה עם",
        "בואו נדבר על",
        "נמשיך עם",
        "ועכשיו נעבור ל",
    ]

    for i, s in enumerate(valid_stocks):
        hebrew = s.get('hebrew', s['ticker'])
        ticker = s['ticker']
        price = s.get('price', '?')
        currency = s.get('currency', '$')
        change = s.get('change_pct', 0)
        direction = "עלתה" if change > 0 else "ירדה" if change < 0 else "נסחרת ללא שינוי"

        section = ""

        # Transition
        if i > 0:
            t = transitions[i % len(transitions)]
            if t.endswith("ל") or t.endswith("על") or t.endswith("עם"):
                section += f"{t}{hebrew}. "
            else:
                section += f"{t} {hebrew}. "

        # Price and change — clear and factual with natural tone
        if abs(change) > 0.5:
            section += f"{hebrew} {direction} {abs(change):.1f} אחוז ונסחרת ב-{currency}{price}. "
        else:
            section += f"{hebrew} נסחרת ב-{currency}{price}, כמעט ללא שינוי. "

        # News summary — the core content
        ns = s.get('news_summary', {})
        if ns.get('has_news') and ns.get('summary'):
            summary = ns['summary'].strip()
            summary = summary.replace('**', '').replace('*', '').replace('#', '')
            section += summary + " "

        # Analyst target if significant gap
        if s.get('target_mean') and s.get('price'):
            upside = round((s['target_mean'] - s['price']) / s['price'] * 100, 1)
            if abs(upside) > 20:
                section += f"יעד האנליסטים עומד על {currency}{s['target_mean']}, פער של {upside:+.0f} אחוז מהמחיר הנוכחי. "

        stock_sections.append(section)

    # --- Step 3: Use Groq for a closing message ---
    closing_prompt = f"""כתוב סיום קצר לפודקאסט כלכלי יומי (40-70 מילים).

מניות שדיברנו עליהן:
{chr(10).join(f'- {s.get("hebrew", s["ticker"])}: {s.get("change_pct", 0):+.1f}%' for s in valid_stocks[:5])}

כללים:
- תובנה אחת מסכמת ליום — מה הנקודה המרכזית
- סיים עם: "זהו להיום. תודה שהאזנתם, ונשמע שוב מחר."
- טון מקצועי וחם, כמו מגיש פודקאסט טוב
- אסור סלנג. אסור "יאללה", "אחי". אסור כוכביות/מספור/אנגלית"""

    closing = _llm_chat([{"role": "user", "content": closing_prompt}], max_tokens=200)

    # --- Step 4: Assemble the full script ---
    script = intro.strip() + "\n\n"
    script += "\n\n".join(stock_sections)
    script += "\n\n" + closing.strip()

    # Final cleanup
    script = script.replace('**', '').replace('*', '').replace('#', '').replace('[', '').replace(']', '')

    print(f"  ✅ Briefing script generated (hybrid) ({len(script)} chars)")
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


def _prepare_text_for_tts(text):
    """Prepare text for better Hebrew TTS pronunciation."""
    import re

    # Stock ticker pronunciation map — how to say them in Hebrew
    ticker_pronunciation = {
        'META': 'מטא',
        'MSFT': 'מייקרוסופט',
        'SOFI': 'סופי',
        'ADBE': 'אדובי',
        'AMZN': 'אמזון',
        'PANW': 'פאלו אלטו',
        'ONON': 'און',
        'IREN': 'אירן',
        'MELI': 'מרקדו ליברה',
        'NVDA': 'אנבידיה',
        'GOOGL': 'גוגל',
        'S&P 500': 'אס אנד פי חמש מאות',
        'S&P': 'אס אנד פי',
        'VIX': 'ויקס',
        'CNBC': 'סי אן בי סי',
        'IBM': 'איי בי אם',
        'AMD': 'איי אם די',
        'GPU': 'ג\'י פי יו',
        'AI': 'איי איי',
        'USD': 'דולר',
        'ILS': 'שקל',
    }

    # Replace English tickers/terms with Hebrew pronunciation
    for eng, heb in ticker_pronunciation.items():
        text = text.replace(eng, heb)

    # Dollar amounts: $34.77 → "34.77 דולר"
    text = re.sub(r'\$(\d+\.?\d*)', r'\1 דולר', text)

    # Shekel amounts: ₪3.13 → "3.13 שקלים"
    text = re.sub(r'₪(\d+\.?\d*)', r'\1 שקלים', text)

    # Percentage: +2.0% → "פלוס 2 אחוז"
    text = re.sub(r'\+(\d+\.?\d*)%', r'פלוס \1 אחוז', text)
    text = re.sub(r'-(\d+\.?\d*)%', r'מינוס \1 אחוז', text)
    text = re.sub(r'(\d+\.?\d*)%', r'\1 אחוז', text)

    # Clean up markdown artifacts
    text = text.replace('**', '').replace('*', '').replace('#', '')
    text = text.replace('[', '').replace(']', '')

    # Clean up double periods
    text = text.replace('..', '.')

    return text


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


def text_to_speech(text, output_path):
    """Convert text to speech — Edge TTS with niqqud, falls back to gTTS."""
    # Step 1: Preprocess (replace tickers, $, % etc)
    text = _prepare_text_for_tts(text)

    # Step 2: Add niqqud for correct pronunciation
    text = _add_niqqud(text)

    # Step 3: Try Edge TTS with plain niqqud text
    try:
        import asyncio
        import edge_tts

        voice = "he-IL-AvriNeural"

        async def _generate():
            communicate = edge_tts.Communicate(text, voice, rate="-5%", pitch="-2Hz")
            await communicate.save(str(output_path))

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(lambda: asyncio.run(_generate())).result(timeout=120)
            else:
                loop.run_until_complete(_generate())
        except RuntimeError:
            asyncio.run(_generate())

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  🔊 Audio saved (Edge TTS + niqqud): {output_path.name} ({size_kb:.0f} KB)")
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
        global PORTFOLIO
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
    # Allow custom tickers from command line
    if len(sys.argv) > 1:
        tickers = [t.strip() for t in ' '.join(sys.argv[1:]).replace(',', ' ').split()]
        run(tickers)
    else:
        run()
