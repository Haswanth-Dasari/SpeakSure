"""Web API tests for resume upload and resume-based interview planning."""

from __future__ import annotations

import json
from pathlib import Path

from test_web_api import _client

from src.speaksure.llm_client import LLMResult

RESUME_TEXT = (
    "Haswanth Dasari\nhaswanth@example.com +91 98765 43210\n"
    "Final-year B.Tech student in computer science.\n"
    "Skills: Python, FastAPI, spaCy\n"
    "Projects: SpeakSure - a spoken interview coach that transcribes answers and writes a report.\n"
)

PROFILE_REPLY = json.dumps(
    {
        "headline": "Final-year B.Tech student",
        "education": [],
        "skills": ["Python", "FastAPI"],
        "projects": [{"name": "SpeakSure", "summary": "A spoken interview coach.", "technologies": []}],
        "experience": [],
        "achievements": [],
    }
)


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, messages, **kwargs):
        self.prompts.append(messages[1]["content"])
        return LLMResult(self.reply, "fake", 0.1, None, None, "stop")


def upload(client, data: bytes, name: str = "cv.txt"):
    return client.post(
        "/api/resume",
        content=data,
        headers={"content-type": "application/octet-stream", "x-filename": name},
    )


def test_upload_without_llm_key_returns_empty_profile_and_warning(tmp_path: Path) -> None:
    client, _stt, _tts, _created = _client(tmp_path, llm_configured=False)
    response = upload(client, RESUME_TEXT.encode())
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["skills"] == [] and "by hand" in body["warning"]


def test_upload_returns_verified_profile_and_redacts_contacts(tmp_path: Path) -> None:
    client, _stt, _tts, created = _client(tmp_path, llm_configured=True)
    llm = FakeLLM(PROFILE_REPLY)
    created["analysis_llm"] = llm
    response = upload(client, RESUME_TEXT.encode())
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["skills"] == ["Python", "FastAPI"]
    assert body["profile"]["projects"][0]["name"] == "SpeakSure"
    assert body["info"]["emails"] == 1 and body["info"]["phones"] == 1
    assert "@" not in llm.prompts[0] and "98765" not in llm.prompts[0]


def test_upload_rejects_unsupported_and_empty_files(tmp_path: Path) -> None:
    client, _stt, _tts, _created = _client(tmp_path, llm_configured=True)
    assert upload(client, b"x" * 200, "cv.exe").status_code == 422
    empty = upload(client, b"", "cv.txt")
    assert empty.status_code == 422 and empty.json()["error"] == "resume_unreadable"


def test_create_session_with_profile_plans_resume_questions(tmp_path: Path) -> None:
    client, _stt, _tts, created = _client(tmp_path)
    response = client.post(
        "/api/sessions",
        json={
            "questions": 5,
            "candidate_name": "Haswanth",
            "resume_profile": {
                "headline": "Student",
                "skills": ["Python"],
                "projects": [{"name": "SpeakSure", "summary": "A spoken interview coach."}],
            },
            "client": {"input_device_label": "Mic"},
        },
    )
    assert response.status_code == 200
    plan = created["interviewer"].plan
    resume_ids = [q.id for q in plan.questions if q.id.startswith("resume_")]
    assert len(plan.questions) == 5 and len(resume_ids) == 2
    assert plan.questions[0].id == "intro" and "SpeakSure" in plan.background
    assert plan.followup_on_last_question is True
    entry = client.app.state.registry.get(response.json()["session_id"])
    metadata = entry.session.metadata
    assert metadata["resume_used"] is True
    assert set(resume_ids) <= {item["id"] for item in metadata["custom_questions"]}
    assert plan.natural_style is True and plan.deep_dive_followups == 2
    assert metadata["resume_profile"]["skills"] == ["Python"]


def test_create_session_without_profile_is_unchanged(tmp_path: Path) -> None:
    client, _stt, _tts, created = _client(tmp_path)
    response = client.post("/api/sessions", json={"questions": 3, "client": {"input_device_label": "Mic"}})
    assert response.status_code == 200
    plan = created["interviewer"].plan
    assert plan.background == "" and plan.followup_on_last_question is False
    entry = client.app.state.registry.get(response.json()["session_id"])
    assert "resume_used" not in entry.session.metadata
    assert "resume_profile" not in entry.session.metadata
    assert plan.natural_style is True and plan.deep_dive_followups == 1


def test_blank_or_invalid_profiles_are_handled(tmp_path: Path) -> None:
    client, _stt, _tts, created = _client(tmp_path)
    blank = client.post(
        "/api/sessions",
        json={"questions": 3, "resume_profile": {"skills": [], "projects": []}, "client": {"input_device_label": "Mic"}},
    )
    assert blank.status_code == 200 and created["interviewer"].plan.background == ""
    invalid = client.post(
        "/api/sessions",
        json={"questions": 3, "resume_profile": "not an object", "client": {"input_device_label": "Mic"}},
    )
    assert invalid.status_code == 422


def test_resume_terms_are_added_to_the_speech_vocabulary_after_the_users_own_hints(tmp_path: Path) -> None:
    client, _stt, _tts, _created = _client(tmp_path)
    response = client.post(
        "/api/sessions",
        json={
            "questions": 3,
            "vocabulary_hints": "Haswanth, speaksure",
            "resume_profile": {
                "skills": ["Python", "FastAPI"],
                "projects": [{"name": "SpeakSure", "summary": "A spoken interview coach."}],
            },
            "client": {"input_device_label": "Mic"},
        },
    )
    assert response.status_code == 200
    entry = client.app.state.registry.get(response.json()["session_id"])
    assert entry.session.metadata["vocabulary_hints"] == ["Haswanth", "speaksure", "FastAPI"]


def test_no_resume_means_no_extra_vocabulary(tmp_path: Path) -> None:
    client, _stt, _tts, _created = _client(tmp_path)
    response = client.post(
        "/api/sessions",
        json={"questions": 3, "client": {"input_device_label": "Mic"}},
    )
    entry = client.app.state.registry.get(response.json()["session_id"])
    assert "vocabulary_hints" not in entry.session.metadata
