import time

from src.utils.logger import logger


def call_with_retry(llm, prompt, retries=5, base_delay=5.0):

    # Groq's free tier answers HTTP 429 when we go over the per-minute
    # limits. Wait 5s, 10s, 20s, ... and try again (exponential backoff).
    for attempt in range(retries):

        try:
            response = llm.invoke(prompt)

            text = response.content if hasattr(response, "content") else str(response)

            return text.strip()

        except Exception as e:

            if attempt == retries - 1:
                raise

            delay = base_delay * (2 ** attempt)

            logger.warning(f"LLM call failed ({e}); retrying in {delay:.0f}s")

            time.sleep(delay)
