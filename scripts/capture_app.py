"""Screenshot the running Streamlit app into docs/images/.

The README embeds these, so they need to be retaken whenever the app or the
underlying panel changes. Boots the app on a scratch port, drives it with
Playwright, and shuts it down again.

    python -m scripts.capture_app

Requires `pip install playwright` and `python -m playwright install chromium`.
"""
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"

# (url path, output file, viewport height)
#
# Heights are fixed per page rather than measured. Streamlit renders into its
# own scroll container and lazily mounts anything below the fold, so at load
# time the DOM genuinely does not know how tall the page will be -- measuring
# it undercounts every page with a chart. The viewport height *is* what
# decides how much lands in the shot, so these are tuned by eye; re-tune if a
# page grows.
PAGES = [
    ("", "app_home.png", 1200),
    ("Data_Explorer", "app_data_explorer.png", 1560),
    ("DiD_Estimator", "app_did_estimator.png", 1240),
    ("Method_Comparison", "app_method_comparison.png", 1430),
]

WIDTH = 1440


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _wait_until_idle(page, timeout=180_000):
    """Block until Streamlit's running indicator clears.

    `networkidle` fires while the script is still executing -- the status
    widget in the top right is the only reliable signal that the rerun has
    finished. The Method Comparison page runs every estimator, so this can
    legitimately take a minute.
    """
    widget = '[data-testid="stStatusWidget"]'
    try:
        # It may already have come and gone; a miss here is not an error.
        page.wait_for_selector(widget, state="visible", timeout=5_000)
    except Exception:
        pass
    page.wait_for_selector(widget, state="detached", timeout=timeout)


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit(
            "playwright is not installed. Run:\n"
            "  pip install playwright && python -m playwright install chromium"
        )
    if shutil.which("streamlit") is None:
        sys.exit("streamlit is not on PATH; activate the project venv first.")

    IMAGES.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    server = subprocess.Popen(
        [
            "streamlit", "run", str(ROOT / "app" / "Home.py"),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_server(port):
            sys.exit(f"streamlit did not come up on port {port}")

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for path, filename, height in PAGES:
                page = browser.new_page(
                    viewport={"width": WIDTH, "height": height},
                    device_scale_factor=2,
                )
                page.goto(f"http://127.0.0.1:{port}/{path}", wait_until="networkidle")
                _wait_until_idle(page)
                time.sleep(3)  # let plotly finish laying out
                page.screenshot(path=str(IMAGES / filename))
                page.close()
                print(f"  wrote {IMAGES / filename} ({WIDTH}x{height})")
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    print("done")


if __name__ == "__main__":
    main()
