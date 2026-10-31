"""Curated HR interview questions and reproducible question selection."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BankQuestion:
    """An HR interview question and the answer details it should elicit."""

    id: str
    topic: str
    text: str
    rubric_hints: tuple[str, ...]


HR_QUESTIONS: tuple[BankQuestion, ...] = (
    BankQuestion(
        id="intro",
        topic="background",
        text="Tell me about yourself.",
        rubric_hints=(
            "current background (education or work)",
            "key skills",
            "one relevant achievement",
            "why this role",
        ),
    ),
    BankQuestion(
        id="motivation",
        topic="motivation",
        text="Why are you interested in this role?",
        rubric_hints=(
            "specific reasons",
            "knowledge of the role or company",
            "link between own skills and the role",
        ),
    ),
    BankQuestion(
        id="strength",
        topic="strengths",
        text="What would you say is your biggest strength, and can you give an example?",
        rubric_hints=("a named strength", "a concrete example", "the outcome"),
    ),
    BankQuestion(
        id="weakness",
        topic="development",
        text="What is a weakness you are working to improve?",
        rubric_hints=(
            "a genuine weakness",
            "self-awareness",
            "concrete steps taken",
            "progress so far",
        ),
    ),
    BankQuestion(
        id="teamwork",
        topic="teamwork",
        text="Tell me about a time you worked in a team to get something done.",
        rubric_hints=("situation", "own task", "own actions", "result (STAR)"),
    ),
    BankQuestion(
        id="conflict",
        topic="conflict",
        text="Describe a disagreement with a teammate and how you handled it.",
        rubric_hints=(
            "situation",
            "respectful approach",
            "resolution",
            "lesson (STAR)",
        ),
    ),
    BankQuestion(
        id="challenge",
        topic="problem solving",
        text="Tell me about a difficult problem you faced and how you solved it.",
        rubric_hints=(
            "the problem",
            "steps taken",
            "own contribution",
            "result (STAR)",
        ),
    ),
    BankQuestion(
        id="mistake",
        topic="learning from mistakes",
        text="Tell me about a time you made a mistake. What did you learn?",
        rubric_hints=(
            "ownership",
            "what happened",
            "lesson",
            "what changed afterwards",
        ),
    ),
    BankQuestion(
        id="pressure",
        topic="pressure",
        text="How do you handle tight deadlines or pressure?",
        rubric_hints=("a concrete strategy", "an example", "outcome"),
    ),
    BankQuestion(
        id="goals",
        topic="career goals",
        text="Where do you see yourself in three to five years?",
        rubric_hints=("realistic goals aligned with the role", "a learning plan"),
    ),
)


def select_questions(count: int, seed: int | None = None) -> tuple[BankQuestion, ...]:
    """Select a deterministic, intro-first subset of distinct HR questions."""
    if not 1 <= count <= len(HR_QUESTIONS):
        raise ValueError(f"count must be from 1 to {len(HR_QUESTIONS)}.")
    remaining = HR_QUESTIONS[1:]
    selected = random.Random(seed).sample(remaining, count - 1)
    selected_ids = {question.id for question in selected}
    return (HR_QUESTIONS[0],) + tuple(
        question for question in remaining if question.id in selected_ids
    )


def question_to_dict(question: BankQuestion) -> dict[str, object]:
    """Serialize a question so it can be stored in session metadata."""
    return {
        "id": question.id,
        "topic": question.topic,
        "text": question.text,
        "rubric_hints": list(question.rubric_hints),
    }


def bank_for_session(metadata: object) -> dict[str, BankQuestion]:
    """Return the HR bank plus any custom (for example resume) questions of a session.

    Custom questions are stored in ``metadata["custom_questions"]`` so that the
    analysis and the report can rate answers against the right hints.
    """
    bank = {question.id: question for question in HR_QUESTIONS}
    custom = metadata.get("custom_questions") if isinstance(metadata, dict) else None
    if not isinstance(custom, list):
        return bank
    for item in custom:
        if not isinstance(item, dict):
            continue
        question_id = item.get("id")
        text = item.get("text")
        hints = item.get("rubric_hints")
        if (
            isinstance(question_id, str)
            and question_id
            and question_id not in bank
            and isinstance(text, str)
            and text
            and isinstance(hints, list)
            and hints
            and all(isinstance(hint, str) for hint in hints)
        ):
            topic = item.get("topic")
            bank[question_id] = BankQuestion(
                id=question_id,
                topic=topic if isinstance(topic, str) and topic else "resume",
                text=text,
                rubric_hints=tuple(hints),
            )
    return bank
