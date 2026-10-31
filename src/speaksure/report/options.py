"""Report display options that do not change the saved analysis."""

from __future__ import annotations

import copy
import os
from typing import Any


def rule_grammar_visible() -> bool:
    """Return whether the rule-based grammar section appears in reports.

    The web app hides it by default: on real spoken transcripts its flags were
    almost all false alarms (see docs/results/e2_grammar.md). Set
    SHOW_RULE_GRAMMAR=1 to show it again.
    """
    return os.environ.get("SHOW_RULE_GRAMMAR", "1").strip().lower() in {"1", "true", "yes"}


def without_rule_grammar(session_data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a session dict for rendering with rule grammar removed.

    Only the report input changes; the saved analysis file keeps the grammar
    results so they can still be evaluated.
    """
    data = copy.deepcopy(session_data)
    for turn in data.get("turns", []):
        analysis = turn.get("analysis") if isinstance(turn, dict) else None
        if isinstance(analysis, dict):
            analysis.pop("grammar", None)
    analysis_block = data.get("analysis")
    if isinstance(analysis_block, dict):
        analysis_block["rule_grammar_hidden"] = True
    return data
