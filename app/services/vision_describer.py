"""Vision LLM service for describing PDF page screenshots.

Uses OpenAI GPT-4o-mini vision to generate natural language descriptions
of UI screenshots in Bottlecapps tutorial PDFs.
"""

import asyncio
import base64
import logging
import random

from openai import AsyncOpenAI, RateLimitError

logger = logging.getLogger(__name__)

VISION_PROMPT_TEMPLATE = (
    "This is page {page_num} of a Bottlecapps admin tutorial: '{doc_title}'.\n"
    "Describe what the screenshot shows. Identify:\n"
    "1) What admin screen or section is displayed\n"
    "2) What buttons, fields, menus, or settings are visible\n"
    "3) What elements are highlighted, circled, or pointed to with arrows\n"
    "4) What action the user is being instructed to perform\n"
    "Write a factual, concise description for a knowledge base.\n"
    "If this page is mostly text with no screenshots, reply with exactly: NO_SCREENSHOT"
)

# Rate limit settings - conservative to avoid 429 errors
MAX_CONCURRENT_REQUESTS = 2  # Reduced from 5 to avoid rate limits
MAX_RETRIES = 5  # More retries for rate limit errors
BASE_DELAY_SECONDS = 2.0  # Base delay between requests
MAX_BACKOFF_SECONDS = 60.0  # Maximum backoff wait time


async def describe_page(
    page_image_bytes: bytes,
    doc_title: str,
    page_num: int,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Describe a single PDF page screenshot using a vision LLM.

    Includes exponential backoff retry logic for rate limit errors.

    Args:
        page_image_bytes: PNG image bytes of the rendered page.
        doc_title: Title of the source PDF document.
        page_num: Page number (0-indexed).
        api_key: OpenAI API key.
        model: Vision model to use.

    Returns:
        Natural language description of the screenshot, or empty string
        if the page has no meaningful screenshots.
    """
    client = AsyncOpenAI(api_key=api_key)
    b64_image = base64.b64encode(page_image_bytes).decode("utf-8")

    prompt = VISION_PROMPT_TEMPLATE.format(
        page_num=page_num + 1,  # Human-readable 1-indexed
        doc_title=doc_title,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )

            description = response.choices[0].message.content.strip()

            # Filter out pages that are purely text
            if description == "NO_SCREENSHOT" or "NO_SCREENSHOT" in description:
                logger.debug(f"Page {page_num + 1} of '{doc_title}': no screenshot detected")
                return ""

            logger.debug(f"Page {page_num + 1} of '{doc_title}': description generated ({len(description)} chars)")
            return description

        except RateLimitError as e:
            # Exponential backoff with jitter for rate limit errors
            if attempt < MAX_RETRIES - 1:
                backoff = min(BASE_DELAY_SECONDS * (2 ** attempt), MAX_BACKOFF_SECONDS)
                jitter = random.uniform(0, backoff * 0.5)
                wait_time = backoff + jitter
                logger.warning(
                    f"Rate limit hit for page {page_num + 1} of '{doc_title}', "
                    f"retry {attempt + 1}/{MAX_RETRIES} in {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries for page {page_num + 1} of '{doc_title}': {e}")
                return ""

        except Exception as e:
            logger.error(f"Error describing page {page_num + 1} of '{doc_title}': {e}")
            return ""

    return ""


async def describe_all_pages(
    page_images: dict[int, bytes],
    doc_title: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> dict[int, str]:
    """Describe all pages of a PDF with rate limiting and backoff.

    Processes pages with limited concurrency and delays between requests
    to respect OpenAI API rate limits.

    Args:
        page_images: Dictionary mapping page number to PNG image bytes.
        doc_title: Title of the source PDF document.
        api_key: OpenAI API key.
        model: Vision model to use.

    Returns:
        Dictionary mapping page number to description string.
        Pages without screenshots map to empty string.
    """
    if not page_images:
        return {}

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def _bounded_describe(page_num: int, img_bytes: bytes) -> tuple[int, str]:
        async with semaphore:
            # Small delay before each request to spread load
            await asyncio.sleep(random.uniform(0.5, 1.5))
            desc = await describe_page(img_bytes, doc_title, page_num, api_key, model)
            return page_num, desc

    tasks = [
        _bounded_describe(page_num, img_bytes)
        for page_num, img_bytes in sorted(page_images.items())
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    descriptions: dict[int, str] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Vision description task failed: {result}")
            continue
        page_num, desc = result
        descriptions[page_num] = desc

    described_count = sum(1 for d in descriptions.values() if d)
    logger.info(
        f"Described {described_count}/{len(page_images)} pages with screenshots for '{doc_title}'"
    )

    return descriptions
