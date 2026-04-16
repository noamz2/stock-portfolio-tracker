"""Refresh NotebookLM auth cookies and push to Supabase Storage.

Run this script from your Mac every ~3 months when cookies expire,
or whenever you see "Authentication expired" / "Missing required cookies" errors.

Requirements:
  - Chrome must be open and logged into Google/NotebookLM
  - Chrome must be running with --remote-debugging-port=9222
    (or use the start_chrome_debug.sh helper)
  - SUPABASE_URL and SUPABASE_SERVICE_KEY in .env

Usage:
  python3 refresh_notebooklm_auth.py
"""

import asyncio
import copy
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

SUPABASE_URL      = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY      = os.environ.get("SUPABASE_SERVICE_KEY", "")
CHROME_PORT       = int(os.environ.get("NOTEBOOKLM_CHROME_PORT", "9222"))
LOCAL_STORAGE     = os.path.expanduser("~/.notebooklm/storage_state.json")
SUPABASE_BUCKET   = "app-config"
SUPABASE_OBJECT   = "notebooklm_storage_state.json"


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


# ── Step 1b: patch domain — copy .google.co.il → .google.com ──────────────

def patch_google_com_cookies(state: dict) -> dict:
    """Israel locale stores SID etc. under .google.co.il, not .google.com.
    notebooklm-py requires .google.com. Copy missing cookies across."""
    cookies = state.get("cookies", [])
    google_com_names = {c.get("name") for c in cookies if c.get("domain") == ".google.com"}
    il_cookies = [c for c in cookies if c.get("domain") == ".google.co.il"]

    added = []
    for c in il_cookies:
        name = c.get("name", "")
        if name not in google_com_names:
            new_c = copy.deepcopy(c)
            new_c["domain"] = ".google.com"
            cookies.append(new_c)
            added.append(name)

    if added:
        print(f"  Patched: copied {len(added)} cookies from .google.co.il → .google.com")
        print(f"           ({', '.join(added)})")
    else:
        print("  No patch needed — .google.com cookies already present")

    # Also update the local file with the patched version
    with open(LOCAL_STORAGE, "w") as f:
        json.dump(state, f)
    return state


# ── Step 2: push to Supabase Storage ───────────────────────────────────────

def push_to_supabase(auth_json: str) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("\n  Supabase credentials not configured.")
        print("  Add to .env:")
        print("    SUPABASE_URL=https://xxx.supabase.co")
        print("    SUPABASE_SERVICE_KEY=eyJ...")
        return False

    import urllib.request
    import urllib.error

    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{SUPABASE_OBJECT}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
    }

    # Try PUT (upsert) first
    data = auth_json.encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        print(f"  Uploaded to Supabase Storage: {SUPABASE_BUCKET}/{SUPABASE_OBJECT}")
        print(f"  Size: {len(auth_json):,} chars | Object ID: {result.get('Id','?')}")
        return True
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        # If object doesn't exist yet, try POST
        if e.code == 404:
            req2 = urllib.request.Request(url.replace("/object/", "/object/"), data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req2) as resp2:
                    json.loads(resp2.read())
                print(f"  Created in Supabase Storage: {SUPABASE_BUCKET}/{SUPABASE_OBJECT}")
                return True
            except Exception as e2:
                print(f"  ERROR creating object: {e2}")
                return False
        print(f"  ERROR uploading to Supabase: {e.code} {e.reason}")
        print(f"  Response: {body[:300]}")
        return False


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 55)
    print("  NotebookLM Auth Refresh")
    print("=" * 55)

    print("\n[1/3] Extracting cookies from Chrome...")
    state = await extract_cookies()

    print("\n[2/3] Patching cookies for notebooklm-py compatibility...")
    state = patch_google_com_cookies(state)
    auth_json = json.dumps(state)
    print(f"  Auth JSON size: {len(auth_json):,} chars")

    print("\n[3/3] Uploading to Supabase Storage...")
    ok = push_to_supabase(auth_json)

    print()
    if ok:
        print("Done. The server will pick up the new cookies on its next podcast request.")
        print("Next refresh: in ~3 months (or when you see 'Authentication expired')")
    else:
        print("Cookies saved locally only. Upload to Supabase manually if needed.")


if __name__ == "__main__":
    asyncio.run(main())
