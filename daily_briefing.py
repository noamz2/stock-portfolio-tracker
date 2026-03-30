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

        # RSI
        hist = sp.history(period='1mo', interval='1d')
        sp_rsi = None
        if not hist.empty and len(hist) > 14:
            closes = hist['Close'].values
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_g = np.mean(gains[-14:])
            avg_l = np.mean(losses[-14:])
            if avg_l > 0:
                sp_rsi = round(100 - (100 / (1 + avg_g / avg_l)), 1)

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

    # RSI
    hist = t.history(period='1mo', interval='1d')
    rsi = None
    if not hist.empty and len(hist) > 14:
        closes = hist['Close'].values
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_g = np.mean(gains[-14:])
        avg_l = np.mean(losses[-14:])
        if avg_l > 0:
            rsi = round(100 - (100 / (1 + avg_g / avg_l)), 1)

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
        'w52_high': round(w52_high / div, 2) if w52_high else None,
        'w52_low': round(w52_low / div, 2) if w52_low else None,
        'pct_from_high': pct_from_high,
        'target_mean': round(target_mean / div, 2) if target_mean else None,
        'target_low': round(target_low / div, 2) if target_low else None,
        'target_high': round(target_high / div, 2) if target_high else None,
        'recommendation': recommendation,
        'news': news,
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

        # RSI + 52W
        sub = []
        if s.get('rsi'):
            rsi_txt = 'Oversold' if s['rsi'] < 30 else 'Overbought' if s['rsi'] > 70 else ''
            sub.append(f"RSI: {s['rsi']}{' (' + rsi_txt + ')' if rsi_txt else ''}")
        if s.get('pct_from_high'):
            sub.append(f"מרחק מ-52W High: {s['pct_from_high']:+.1f}%")
        if sub:
            lines.append(f"    {' | '.join(sub)}")

        # Analyst target
        if s.get('target_mean'):
            upside = round((s['target_mean'] - s['price']) / s['price'] * 100, 1)
            lines.append(f"    🎯 יעד אנליסטים: {s['currency']}{s['target_mean']} ({upside:+.1f}% upside) | המלצה: {s.get('recommendation', '—')}")

        # News
        if s.get('news'):
            lines.append(f"    📰 חדשות:")
            for n in s['news'][:3]:
                lines.append(f"       • {n['title']}")

    lines.append("")
    lines.append("─" * 40)
    lines.append("⚠️  אין זו המלצת השקעה. נא להתייעץ עם יועץ מורשה.")
    lines.append(f"נוצר: {datetime.now().strftime('%H:%M %d/%m/%Y')}")

    return '\n'.join(lines)


def generate_podcast_script(data):
    """Generate Hebrew podcast script from data."""
    m = data['macro']
    stocks = data['stocks']
    valid_stocks = [s for s in stocks if 'error' not in s]
    date = datetime.fromisoformat(data['date']).strftime('%d/%m/%Y')

    fg = m.get('fear_greed', {})
    sp = m.get('sp500', {})

    script = f"""שלום וברוכים הבאים לסיכום היומי של תיק ההשקעות, תאריך {date}.
זו אינה המלצת השקעה.

נתחיל בתמונת המאקרו:
"""

    # Macro section
    if sp.get('price'):
        script += f"מדד ה-S&P 500 עומד על {sp['price']:,.0f} נקודות, שינוי של {sp.get('change_pct', 0):+.1f} אחוז. "

    if fg.get('score') is not None:
        score = fg['score']
        if score < 20:
            script += f"מדד הפחד והתאוות בצע עומד על {score}, שזה פחד קיצוני. היסטורית, רמות כאלה נוטות להיות הזדמנויות קנייה לטווח הארוך. "
        elif score > 75:
            script += f"מדד הפחד והתאוות בצע עומד על {score}, שזה אזור של תאוות בצע. כדאי לשקול לקחת חלק מהרווחים. "
        else:
            script += f"מדד הפחד והתאוות בצע עומד על {score}. "

    if m.get('vix'):
        if m['vix'] > 25:
            script += f"מדד ה-VIX גבוה על {m['vix']:.0f}, מה שמצביע על תנודתיות גבוהה ופחד בשוק. "
        else:
            script += f"מדד ה-VIX על {m['vix']:.0f}. "

    if sp.get('rsi') and sp['rsi'] < 30:
        script += f"ה-RSI של ה-S&P 500 עומד על {sp['rsi']:.0f}, שזה אזור של oversold. "

    if m.get('usd_ils'):
        script += f"שער הדולר עומד על {m['usd_ils']:.2f} שקלים. "

    # Stocks section
    winners = [s for s in valid_stocks if s.get('change_pct', 0) > 1]
    losers = [s for s in valid_stocks if s.get('change_pct', 0) < -1]

    script += "\n\nעכשיו לניתוח המניות בתיק.\n"

    if losers:
        losers.sort(key=lambda s: s['change_pct'])
        script += "\nהמניות שירדו בולטות: "
        for s in losers[:3]:
            script += f"{s['hebrew']} ירדה {abs(s['change_pct']):.1f} אחוז ועומדת על {s['currency']}{s['price']}. "
            if s.get('pct_from_high') and s['pct_from_high'] < -20:
                script += f"המניה נמצאת {abs(s['pct_from_high']):.0f} אחוז מתחת לשיא של 52 שבועות. "
            if s.get('rsi') and s['rsi'] < 30:
                script += f"ה-RSI נמצא באזור oversold על {s['rsi']:.0f}. "

    if winners:
        winners.sort(key=lambda s: s['change_pct'], reverse=True)
        script += "\nמהצד החיובי: "
        for s in winners[:3]:
            script += f"{s['hebrew']} עלתה {s['change_pct']:.1f} אחוז ל-{s['currency']}{s['price']}. "

    # Detailed per-stock
    script += "\n\nניתוח מפורט של המניות המרכזיות:\n"

    for s in valid_stocks[:6]:
        script += f"\n{s['hebrew']}: "
        script += f"המחיר עומד על {s['currency']}{s['price']}, שינוי של {s['change_pct']:+.1f} אחוז. "
        if s.get('pe') and s.get('forward_pe'):
            script += f"מכפיל הרווח הנוכחי הוא {s['pe']:.0f} ומכפיל הרווח העתידי {s['forward_pe']:.0f}. "
        if s.get('target_mean'):
            upside = round((s['target_mean'] - s['price']) / s['price'] * 100, 1)
            if upside > 10:
                script += f"יעד האנליסטים הממוצע הוא {s['currency']}{s['target_mean']}, מה שמשקף פוטנציאל עלייה של {upside:.0f} אחוז. "
            elif upside < -5:
                script += f"יעד האנליסטים נמצא מתחת למחיר הנוכחי, מה שמרמז על תמחור יתר. "
        if s.get('news') and s['news']:
            script += f"בחדשות: {s['news'][0]['title']}. "

    # Summary
    script += f"""

לסיכום: """

    if (fg.get('score') or 50) < 20:
        script += "השוק נמצא בפחד קיצוני. היסטורית מצבים כאלה נוטים להיות הזדמנויות, אבל חשוב להישאר ממושמעים ולא למכור בפאניקה. "

    oversold = [s for s in valid_stocks if s.get('rsi') and s['rsi'] < 30]
    if oversold:
        names = ', '.join(s['hebrew'] for s in oversold)
        script += f"המניות {names} נמצאות באזור oversold מבחינה טכנית. "

    big_upside = [s for s in valid_stocks if s.get('target_mean') and ((s['target_mean'] - s['price']) / s['price'] * 100) > 20]
    if big_upside:
        names = ', '.join(s['hebrew'] for s in big_upside)
        script += f"מבחינת פוטנציאל עלייה, {names} מציגות את הפער הגדול ביותר ליעד האנליסטים. "

    script += "\nזה הסיכום להיום. יום מוצלח ותודה שהאזנתם!"

    return script


def text_to_speech(text, output_path):
    """Convert text to speech using gTTS."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='iw', slow=False)
        tts.save(str(output_path))
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  🔊 Audio saved: {output_path.name} ({size_kb:.0f} KB)")
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
