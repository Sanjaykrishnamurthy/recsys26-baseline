"""Null LM module for retrieval-only evaluation.

Returns an empty string for every generation request. Use this when you want
to evaluate only the retrieval stage (predicted_track_ids) without loading
any language model — no GPU, no model download, no latency.

Usage in config:
    lm_type: "none"
"""

from typing import List


class NULL_LM:
    """No-op language model that always returns an empty response.

    Satisfies the same interface as LLAMA_MODEL / OLLAMA_MODEL so it can be
    dropped into CRS_BASELINE without any other changes.
    """

    def __init__(self) -> None:
        print("[INFO] NULL_LM loaded — retrieval-only mode, responses will be empty strings.")

    def response_generation(
        self,
        sys_prompt: str,
        chat_history: list,
        recommend_item: str,
        max_new_tokens: int = 0,
        response_format=None,
    ) -> str:
        """Return an empty string (no generation performed).

        Args:
            sys_prompt: Unused.
            chat_history: Unused.
            recommend_item: Unused.
            max_new_tokens: Unused.
            response_format: Unused.

        Returns:
            Empty string.
        """
        return ""

    def batch_response_generation(
        self,
        sys_prompts: List[str],
        chat_histories: List[list],
        recommend_items: List[str],
        max_new_tokens: int = 0,
    ) -> List[str]:
        """Return empty strings for every item in the batch.

        Args:
            sys_prompts: Unused.
            chat_histories: Unused.
            recommend_items: Unused.
            max_new_tokens: Unused.

        Returns:
            List of empty strings, one per input.
        """
        return ["" for _ in sys_prompts]
