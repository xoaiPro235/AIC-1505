from __future__ import annotations

import logging
import re

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Compact long natural-language queries before SigLIP2 encoding.

    Gemini is used only as a text preprocessor. The final retrieval vector is
    still produced by SigLIP2, keeping it compatible with the Qdrant image
    vectors that were built with SigLIP2.
    """

    def __init__(
        self,
        gemini_api_key: str | None = None,
        text_model_name: str = "gemini-3.5-flash-lite",
        min_chars: int = 160,
        min_words: int = 35,
    ):
        self.text_model_name = text_model_name
        self.min_chars = min_chars
        self.min_words = min_words
        self.client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
        self._cache: dict[str, str] = {}

    def should_rewrite(self, query: str) -> bool:
        if not self.client:
            return False
        clean = query.strip()
        word_count = len(clean.split())
        clause_count = len(re.findall(r"[,.;:。،؛]", clean))
        return len(clean) >= self.min_chars or word_count >= self.min_words or clause_count >= 3

    def rewrite(self, query: str, max_words: int = 45, force: bool = False) -> str:
        clean_query = query.strip()
        if not clean_query:
            return query
        if not force and not self.should_rewrite(clean_query):
            logger.info("Rewrite skipped: query is short enough for SigLIP2.")
            return clean_query
        if clean_query in self._cache:
            return self._cache[clean_query]
        if not self.client:
            return clean_query

        prompt = f"""
You rewrite video retrieval queries for a SigLIP2 text encoder with a short context window.
Convert the input into one concise visual search phrase of at most {max_words} words.
Keep only visible evidence: objects, people, actions, scene, colors, place, OCR/text cues, and temporal moment.
Do not invent details. Prefer clear English visual wording. Return only the rewritten phrase.

Input query:
{clean_query}
""".strip()
        try:
            response = self.client.models.generate_content(
                model=self.text_model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            rewritten = (response.text or "").strip().strip('"').strip("'")
            if not rewritten:
                rewritten = clean_query
        except Exception as exc:
            logger.warning("Query rewrite failed, using original query: %s", exc)
            rewritten = clean_query

        self._cache[clean_query] = rewritten
        logger.info("Rewritten query: %s", rewritten)
        return rewritten
