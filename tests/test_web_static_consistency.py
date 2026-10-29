"""Static checks that catch browser-script typos no Python test would see.

Three real bugs shipped because ``main.js`` read ``app.roomRole`` and
``app.room`` (which live in ``el``) and wrote ``el.analysisStartedAt`` (which
lives in ``app``). These tests fail when a property is used but never defined,
or when the script asks for an element id the page does not contain.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "speaksure" / "web" / "static"
MAIN_JS = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")


def _object_keys(name: str) -> set[str]:
    match = re.search(rf"const {name} = \{{(.*?)\n\}};", MAIN_JS, re.S)
    assert match is not None, f"const {name} object not found in main.js"
    return set(re.findall(r"^\s{2}(\w+):", match.group(1), re.M))


def test_app_properties_are_defined_or_assigned() -> None:
    defined = _object_keys("app") | set(re.findall(r"\bapp\.(\w+)\s*=[^=]", MAIN_JS))
    used = set(re.findall(r"\bapp\.(\w+)", MAIN_JS))
    assert sorted(used - defined) == []


def test_el_properties_are_defined() -> None:
    used = set(re.findall(r"\bel\.(\w+)", MAIN_JS))
    assert sorted(used - _object_keys("el")) == []


def test_element_ids_exist_in_page() -> None:
    ids = set(re.findall(r'getElementById\("([^"]+)"\)', MAIN_JS))
    assert ids, "no getElementById calls found"
    missing = sorted(i for i in ids if f'id="{i}"' not in INDEX_HTML)
    assert missing == []
