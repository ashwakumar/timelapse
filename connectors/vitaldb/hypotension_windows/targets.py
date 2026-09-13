"""Learning targets shared by the TimeNet connector and task card."""

from __future__ import annotations

ANSWER_TEXT = {
    "within_3": "hypotension within 3 minutes",
    "within_5": "hypotension within 5 minutes",
    "within_10": "hypotension within 10 minutes",
    "within_15": "hypotension within 15 minutes",
    "none_within_15": "no hypotension within 15 minutes",
}

QA_PROMPT = (
    "Using only the supplied 20-second signal history and observation masks, predict the tightest "
    "available horizon containing the first onset of sustained MAP below 65 mmHg. Do not infer a "
    "causal mechanism or recommend a drug or dose."
)
