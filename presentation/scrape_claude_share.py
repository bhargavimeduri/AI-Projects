"""
Scrapes a Claude.ai shared conversation and saves the full text to a file.
"""
import asyncio
from playwright.async_api import async_playwright

URL = "https://claude.ai/share/dd273cb6-512f-4132-9707-d5e7c14d3ff8"
OUTPUT = "presentation/Mayank_Blueprint_Raw.txt"

async def scrape():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # headless=False so you can see it
        page = await browser.new_page()

        # Set realistic browser headers to bypass Cloudflare
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        })

        print(f"[INFO] Opening: {URL}")
        await page.goto(URL, timeout=30000)

        # Wait longer for Cloudflare challenge + content to load
        print("[INFO] Waiting for Cloudflare + content (15 sec)...")
        await page.wait_for_timeout(15000)

        # Try to extract all visible text from the conversation
        content = await page.evaluate("""
            () => {
                // Get all text nodes from conversation messages
                const selectors = [
                    '[data-testid="message"]',
                    '.prose',
                    'article',
                    'main',
                    '[class*="message"]',
                    '[class*="content"]',
                    '[class*="conversation"]'
                ];

                let text = '';
                for (const sel of selectors) {
                    const elements = document.querySelectorAll(sel);
                    if (elements.length > 0) {
                        elements.forEach(el => {
                            text += el.innerText + '\\n\\n---\\n\\n';
                        });
                        break;
                    }
                }

                // Fallback: grab all text from body
                if (!text.trim()) {
                    text = document.body.innerText;
                }

                return text;
            }
        """)

        await browser.close()

        if content and len(content.strip()) > 100:
            with open(OUTPUT, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[SUCCESS] Saved {len(content)} characters to {OUTPUT}")
        else:
            print("[WARN] Content too short — page may not have loaded fully.")
            print("Content preview:", content[:500] if content else "(empty)")

asyncio.run(scrape())
