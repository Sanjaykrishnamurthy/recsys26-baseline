"""Ollama-based LM module for local inference without HuggingFace downloads.

Ollama runs models locally via a lightweight server (http://localhost:11434).
It supports Llama, Qwen, Phi, Mistral, Gemma, and more — downloaded from
ollama.com rather than HuggingFace, which avoids corporate SSL restrictions.

Setup:
    brew install ollama
    ollama serve &
    ollama pull llama3.2:1b   # or qwen3:4b, phi4-mini, mistral, etc.

Usage in config:
    lm_type: "ollama:llama3.2:1b"
    # or
    lm_type: "ollama:qwen3:4b"
"""

from __future__ import annotations

import json
from typing import List
import urllib.request
import urllib.error


class OLLAMA_MODEL:
    """LM module that delegates generation to a local Ollama server.

    Attributes:
        model_name: Ollama model tag (e.g. "llama3.2:1b", "qwen3:4b").
        host: Base URL of the Ollama server.
        api_url: Full chat completion endpoint URL.
    """

    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        host: str = "http://localhost:11434",
    ) -> None:
        """Initialise and verify the Ollama server is reachable.

        Args:
            model_name: Ollama model tag to use for generation.
            host: Base URL where the Ollama server is running.

        Raises:
            RuntimeError: If the Ollama server is not running or the model
                          is not available.
        """
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.api_url = f"{self.host}/api/chat"
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Check that the Ollama server is running and the model is available."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            available = [m["name"] for m in data.get("models", [])]
            # Ollama tags can be "llama3.2:1b" or "llama3.2:1b-instruct-q4_K_M"
            matched = any(self.model_name.split(":")[0] in m for m in available)
            if not matched:
                print(
                    f"[WARNING] Model '{self.model_name}' not found in Ollama. "
                    f"Available: {available}\n"
                    f"Run: ollama pull {self.model_name}"
                )
            else:
                print(f"[INFO] Ollama model '{self.model_name}' is ready.")
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                f"Make sure it is running: `ollama serve`\nError: {e}"
            )

    def _build_messages(
        self,
        sys_prompt: str,
        chat_history: list,
        recommend_item: str,
    ) -> List[dict]:
        """Build the message list in Ollama / OpenAI chat format.

        Args:
            sys_prompt: System prompt string.
            chat_history: List of {"role": ..., "content": ...} dicts.
            recommend_item: Top-1 recommended track metadata string.

        Returns:
            List of message dicts ready for the Ollama API.
        """
        messages = [{"role": "system", "content": sys_prompt}]
        messages += chat_history
        messages += [{"role": "assistant", "content": recommend_item}]
        return messages

    def _call_api(self, messages: List[dict], max_new_tokens: int) -> str:
        """Send a chat request to Ollama and return the generated text.

        Args:
            messages: List of chat message dicts.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated response string.
        """
        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_new_tokens},
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        return result["message"]["content"]

    # ------------------------------------------------------------------
    # Public API — matches LLAMA_MODEL interface
    # ------------------------------------------------------------------

    def response_generation(
        self,
        sys_prompt: str,
        chat_history: list,
        recommend_item: str,
        max_new_tokens: int = 256,
        response_format=None,
    ) -> str:
        """Generate a single response via Ollama.

        Args:
            sys_prompt: System prompt string.
            chat_history: Conversation history as list of role/content dicts.
            recommend_item: Top-1 recommended track metadata string.
            max_new_tokens: Maximum tokens to generate.
            response_format: Unused; kept for interface compatibility.

        Returns:
            Generated response string.
        """
        messages = self._build_messages(sys_prompt, chat_history, recommend_item)
        return self._call_api(messages, max_new_tokens)

    def batch_response_generation(
        self,
        sys_prompts: List[str],
        chat_histories: List[list],
        recommend_items: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        """Generate responses for a batch sequentially via Ollama.

        Ollama does not support true parallel batching, so requests are
        sent one at a time. For faster throughput, run multiple Ollama
        processes or use a GPU-accelerated setup.

        Args:
            sys_prompts: List of system prompts.
            chat_histories: List of conversation history lists.
            recommend_items: List of top-1 recommended track metadata strings.
            max_new_tokens: Maximum tokens per response.

        Returns:
            List of generated response strings.
        """
        return [
            self.response_generation(sp, ch, ri, max_new_tokens)
            for sp, ch, ri in zip(sys_prompts, chat_histories, recommend_items)
        ]
