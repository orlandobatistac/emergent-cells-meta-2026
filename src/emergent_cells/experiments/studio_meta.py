"""Optional Meta Model API adapter for the Studio research copilot.

Meta Muse Spark is an existing hosted foundation model. It does not power or
train the local experimental cell network and cannot execute actions.
"""
from __future__ import annotations

import json
import os
from urllib import error, request

MODEL = "muse-spark-1.3"
ENDPOINT = "https://api.meta.ai/v1/chat/completions"


def is_configured() -> bool:
    return bool(os.environ.get("MODEL_API_KEY", "").strip())


def ask_meta_model(message: str, safe_status: dict, *, timeout: int = 35) -> str:
    """Generate a grounded explanation without sharing hidden maze truth or secrets."""
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        return ("Meta Model API is not configured. Set MODEL_API_KEY in the "
                "Studio server environment and restart, or select local Ollama.")
    system = (
        "You are a scientific research assistant for Emergent Cells Studio. "
        "You are an existing Meta Muse Spark language model, NOT a cellular "
        "language model or a decentralized agent. The cell controller has "
        "engineered exploration rules, authored message values and relative "
        "odometry. Do not imply general intelligence, actual multi-PC "
        "distribution or measured electrical energy savings. Differentiate "
        "hypotheses from confirmed experimental evidence. You cannot execute "
        "code, shell commands, browser actions or live simulator tools. "
        "Keep answers brief and in the user's language. "
        "The following user-visible status is observer telemetry, not a "
        "privileged input to the policy: " + json.dumps(safe_status)
    )
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": message[:1000]}],
        "max_tokens": 360,
    }).encode("utf-8")
    req = request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
        result = data["choices"][0]["message"]["content"]
        if not isinstance(result, str) or not result.strip():
            return "Meta Model API returned no textual answer."
        return result.strip()
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return ("Meta Model API request failed or returned an unexpected "
                "response. The local cellular simulator remains available.")
