"""
config.py — Shared configuration for Part 1 and Part 2 pipelines.

Loads settings from environment variables and exposes them as module-level constants.
"""

import os
from groq import Groq

# ── LLM settings ───────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_CONTEXT_CHARS = 12000  # max characters passed to LLM to avoid token overflow

# ── Part 1: codebase path ──────────────────────────────────────────────────────
REPO_PATH = os.environ.get("REPO_PATH", "./mcp-gateway-registry")

# ── Part 2: data paths ─────────────────────────────────────────────────────────
CSV_PATH = os.environ.get("CSV_PATH", "./data/structured/daily_sales.csv")
TEXT_DIR = os.environ.get("TEXT_DIR", "./data/unstructured/")

# ── Groq client ────────────────────────────────────────────────────────────────
def get_client() -> Groq:
    """
    Initialize and return a Groq client using the GROQ_API_KEY environment variable.
    Raises ValueError if the key is not set.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please run: export GROQ_API_KEY='your_key'"
        )
    return Groq(api_key=api_key)
