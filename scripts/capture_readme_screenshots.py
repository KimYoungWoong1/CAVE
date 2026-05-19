"""Capture README-safe Streamlit screenshots with Playwright.

Run Streamlit first:

    streamlit run app.py --server.port 8501

Then run:

    python scripts/capture_readme_screenshots.py

The screenshots use real local pipeline output, but the uploaded media preview is
blurred before capture so the repository does not redistribute dataset images.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import async_playwright


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://localhost:8501"
OUTPUT_DIR = ROOT / "docs" / "assets"

BLUR_MEDIA_CSS = """
[data-testid="stImage"] {
  position: relative !important;
  overflow: hidden !important;
}
[data-testid="stImage"] img {
  filter: blur(24px) brightness(0.82) saturate(0.7) !important;
  transform: scale(1.04) !important;
}
[data-testid="stImage"]::after {
  content: "Media preview blurred for README";
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  background: rgba(15, 23, 42, 0.42);
  font: 700 28px -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif;
  text-align: center;
  letter-spacing: 0;
}
"""


@dataclass(frozen=True)
class CaptureCase:
    name: str
    sample_path: Path
    output_path: Path
    crime_label: str = "딥페이크 성범죄"


CASES = [
    CaptureCase(
        name="ai_suspected",
        sample_path=ROOT
        / "test_data"
        / "genimage"
        / "holdout"
        / "ai"
        / "GLIDE__tiny_genimage_0000831.jpg",
        output_path=OUTPUT_DIR / "cave_screen_ai_suspected.png",
    ),
    CaptureCase(
        name="authentic_likely",
        sample_path=ROOT
        / "test_data"
        / "genimage"
        / "holdout"
        / "real"
        / "Real__tiny_genimage_0000532.jpg",
        output_path=OUTPUT_DIR / "cave_screen_authentic_likely.png",
        crime_label="기본",
    ),
]


async def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            for case in CASES:
                await capture_case(browser, args.url, case)
        finally:
            await browser.close()


async def capture_case(browser, url: str, case: CaptureCase) -> None:
    if not case.sample_path.exists():
        raise FileNotFoundError(
            f"Missing sample for {case.name}: {case.sample_path.relative_to(ROOT)}"
        )

    context = await browser.new_context(
        viewport={"width": 1680, "height": 980},
        device_scale_factor=1,
        locale="ko-KR",
    )
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        if case.crime_label != "딥페이크 성범죄":
            await page.locator('[data-testid="stSelectbox"]').first.click()
            await page.get_by_role("option", name=case.crime_label).click(timeout=10_000)
            await page.wait_for_timeout(800)
        await page.locator("input[type=file]").set_input_files(str(case.sample_path))
        await page.wait_for_timeout(1_500)
        await page.locator("button", has_text="분석 실행").click(timeout=30_000)

        await page.get_by_text("최종 의견").wait_for(timeout=180_000)
        await page.get_by_text("피해·확산 근거").wait_for(timeout=180_000)
        await page.wait_for_timeout(1_200)
        await page.add_style_tag(content=BLUR_MEDIA_CSS)
        await page.screenshot(path=str(case.output_path), full_page=False)
        print(f"Wrote {case.output_path.relative_to(ROOT)}")
    finally:
        await context.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="Running Streamlit URL")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
