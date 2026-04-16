"""One-shot NotebookLM login — opens browser, saves cookies, uploads to Supabase.

Run this whenever you see "Authentication expired" or "Missing required cookies".

Usage:
  python3 notebooklm_login.py
"""

import copy
import json
import os
import sys
import urllib.request

# ── Load .env ──────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

SUPABASE_URL    = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY    = os.environ.get("SUPABASE_SERVICE_KEY", "")
STORAGE_PATH    = os.path.expanduser("~/.notebooklm/storage_state.json")
BROWSER_PROFILE = os.path.expanduser("~/.notebooklm/browser_profile")
BUCKET          = "app-config"
OBJECT          = "notebooklm_storage_state.json"


def patch_google_com_cookies(state: dict) -> dict:
    """Copy SID etc. from .google.co.il → .google.com (Israel locale fix)."""
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
        print(f"  Patched {len(added)} cookies: .google.co.il → .google.com")
    return state


def upload_to_supabase(auth_json: str) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("  WARNING: No Supabase credentials — skipping upload.")
        return False
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{OBJECT}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
    }
    data = auth_json.encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(req) as resp:
            json.loads(resp.read())
        print(f"  Uploaded to Supabase Storage ({len(auth_json):,} chars)")
        return True
    except Exception as e:
        print(f"  ERROR uploading to Supabase: {e}")
        return False


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed.")
        print("Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    os.makedirs(os.path.dirname(STORAGE_PATH), exist_ok=True)
    os.makedirs(BROWSER_PROFILE, exist_ok=True)

    print("=" * 55)
    print("  NotebookLM Login")
    print("=" * 55)
    print("\nOpening browser for Google login...")
    print("Profile:", BROWSER_PROFILE)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=BROWSER_PROFILE,
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--password-store=basic",
            ],
            ignore_default_args=["--enable-automation"],
        )

        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://notebooklm.google.com/")

        print("\n" + "=" * 55)
        print("INSTRUCTIONS:")
        print("  1. Complete the Google login in the browser window")
        print("  2. Navigate to notebooklm.google.com if not there yet")
        print("  3. This window will save and close automatically")
        print("=" * 55)
        print("\nWaiting for you to log in... (up to 5 minutes)")

        # Wait until the user lands on notebooklm.google.com (auto, no Enter needed)
        try:
            page.wait_for_url("*notebooklm.google.com*", timeout=300_000)
        except Exception:
            pass  # timeout — save whatever state we have

        current_url = page.url
        print(f"\nCurrent URL: {current_url}")

        if "notebooklm.google.com" not in current_url:
            print(f"WARNING: Not on NotebookLM (url={current_url[:80]})")
            print("Saving anyway...")

        context.storage_state(path=STORAGE_PATH)
        context.close()

    print(f"\nCookies saved to: {STORAGE_PATH}")

    with open(STORAGE_PATH) as f:
        state = json.load(f)

    print(f"Total cookies: {len(state.get('cookies', []))}")

    print("\nPatching domain cookies...")
    state = patch_google_com_cookies(state)

    auth_json = json.dumps(state)
    with open(STORAGE_PATH, "w") as f:
        f.write(auth_json)

    print("\nUploading to Supabase Storage...")
    ok = upload_to_supabase(auth_json)

    print()
    if ok:
        print("Done! Podcast generation will now work on Render.")
        print("Next refresh: in ~3 months (or when you see auth errors again).")
    else:
        print("Cookies saved locally. Upload to Supabase manually if needed.")


if __name__ == "__main__":
    main()
