"""Refresh NotebookLM auth cookies and push to Render.

Run this script from your Mac every ~3 months when cookies expire,
or whenever you see "Authentication expired" errors in the briefing.

Requirements:
  - Chrome must be open and logged into Google/NotebookLM
  - Chrome must be running with --remote-debugging-port=9222
    (or use the start_chrome_debug.sh helper)
  - RENDER_API_KEY and RENDER_SERVICE_ID in .env

Usage:
  python3 refresh_notebooklm_auth.py
"""

import asyncio
import json
import os
import sys

# ── Load .env ──────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

RENDER_API_KEY    = os.environ.get("RENDER_API_KEY", "")
RENDER_SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "")
CHROME_PORT       = int(os.environ.get("NOTEBOOKLM_CHROME_PORT", "9222"))
LOCAL_STORAGE     = os.path.expanduser("~/.notebooklm/storage_state.json")


# ── Step 1: extract cookies from Chrome ────────────────────────────────────

async def extract_cookies() -> dict:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run: pip install playwright")
        sys.exit(1)

    async with async_playwright() as p:
        try:
            browser = await p.chromium.connect_over_cdp(f"http://localhost:{CHROME_PORT}")
        except Exception as e:
            print(f"ERROR: Cannot connect to Chrome on port {CHROME_PORT}.")
            print("Make sure Chrome is open and running with --remote-debugging-port=9222")
            print(f"Details: {e}")
            sys.exit(1)

        contexts = browser.contexts
        if not contexts:
            print("ERROR: No browser contexts found. Open Chrome and log in to NotebookLM first.")
            sys.exit(1)

        ctx = contexts[0]
        os.makedirs(os.path.dirname(LOCAL_STORAGE), exist_ok=True)
        await ctx.storage_state(path=LOCAL_STORAGE)

        with open(LOCAL_STORAGE) as f:
            state = json.load(f)

    cookies = state.get("cookies", [])
    google_cookies = [c for c in cookies if "google" in c.get("domain", "")]
    print(f"  Extracted {len(cookies)} cookies ({len(google_cookies)} Google)")
    return state


# ── Step 2: push to Render ──────────────────────────────────────────────────

def push_to_render(auth_json: str) -> bool:
    if not RENDER_API_KEY or not RENDER_SERVICE_ID:
        print("\n  Render credentials not configured.")
        print("  Add to .env:")
        print("    RENDER_API_KEY=rnd_xxxx")
        print("    RENDER_SERVICE_ID=srv-xxxx")
        print("\n  Manual fallback — copy this into Render dashboard as NOTEBOOKLM_AUTH_JSON:")
        print("-" * 60)
        # Print a truncated preview so the terminal doesn't flood
        preview = auth_json[:120] + "..." if len(auth_json) > 120 else auth_json
        print(preview)
        print("-" * 60)
        return False

    import urllib.request
    import urllib.error

    headers = {
        "Authorization": f"Bearer {RENDER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Get current env vars so we don't accidentally wipe them
    url_list = f"https://api.render.com/v1/services/{RENDER_SERVICE_ID}/env-vars"
    req = urllib.request.Request(url_list, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            current = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  ERROR fetching current env vars: {e.code} {e.reason}")
        return False

    # Build updated list — keep all existing vars, update/add NOTEBOOKLM_AUTH_JSON
    env_vars = []
    found = False
    for item in current:
        kv = item.get("envVar", item)  # Render wraps in envVar key
        key = kv.get("key", "")
        if key == "NOTEBOOKLM_AUTH_JSON":
            env_vars.append({"key": key, "value": auth_json})
            found = True
        else:
            env_vars.append({"key": key, "value": kv.get("value", "")})

    if not found:
        env_vars.append({"key": "NOTEBOOKLM_AUTH_JSON", "value": auth_json})

    # PUT the updated list
    payload = json.dumps(env_vars).encode()
    req_put = urllib.request.Request(url_list, data=payload, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(req_put) as resp:
            resp.read()
        print(f"  Successfully updated NOTEBOOKLM_AUTH_JSON on Render ({len(env_vars)} env vars total)")
        return True
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  ERROR updating Render: {e.code} {e.reason}")
        print(f"  Response: {body[:300]}")
        return False


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 55)
    print("  NotebookLM Auth Refresh")
    print("=" * 55)

    print("\n[1/2] Extracting cookies from Chrome...")
    state = await extract_cookies()
    auth_json = json.dumps(state)
    print(f"  Auth JSON size: {len(auth_json):,} chars")

    print("\n[2/2] Pushing to Render...")
    ok = push_to_render(auth_json)

    print()
    if ok:
        print("Done. The server will pick up the new cookies on its next request.")
        print("Next refresh: in ~3 months (or when you see 'Authentication expired')")
    else:
        print("Cookies saved locally. Update Render manually if needed.")


if __name__ == "__main__":
    asyncio.run(main())
