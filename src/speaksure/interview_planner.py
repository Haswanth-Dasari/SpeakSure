"""Plan a varied, realistic interview instead of a fixed list of questions.

Real interviews are not identical every time. This planner keeps the familiar
shape (a warm-up first, a closing question last) but draws the middle from a
larger pool, mixes resume questions in at different positions, avoids opening
with a negative question, and varies the wording of the opener. Selection is
reproducible when a seed is given, which keeps tests and experiments exact.
"""

from __future__ import annotations

import random
from typing import Any

from .llm_client import LLMClient
from .question_bank import HR_QUESTIONS, BankQuestion
from .resume_questions import build_resume_questions, resume_question_count

INTRO_VARIANTS = (
    "Tell me about yourself.",
    "To start, could you walk me through your background?",
    "Let's begin with you. Tell me a little about who you are and what you have been working on.",
    "Thanks for joining. Can you start by telling me a bit about yourself?",
)

EXTRA_QUESTIONS: tuple[BankQuestion, ...] = (
    BankQuestion(
        "hire_you",
        "fit",
        "Why should we hire you for this role?",
        ("skills matched to the role", "a concrete example", "what they would add"),
    ),
    BankQuestion(
        "situational_disagree",
        "situational",
        "Imagine you disagree with a senior teammate about a technical decision. What would you do?",
        (
            "listening to the other view first",
            "using evidence or a small test",
            "respectful compromise",
            "escalating only if needed",
        ),
    ),
    BankQuestion(
        "learning",
        "learning",
        "Tell me about something new you learned recently and how you went about learning it.",
        ("what was learned", "the method used", "how it was applied"),
    ),
    BankQuestion(
        "feedback",
        "feedback",
        "Tell me about feedback you received that was hard to hear. What did you do with it?",
        ("the feedback itself", "the reaction", "concrete changes made", "the result"),
    ),
    BankQuestion(
        "proud",
        "achievement",
        "Which piece of work are you most proud of, and why?",
        ("what it was", "own contribution", "why it matters", "the result"),
    ),
    BankQuestion(
        "candidate_questions",
        "closing",
        "Before we finish, what questions do you have for us about the role or the team?",
        (
            "relevant questions about the role or team",
            "shows research or curiosity",
            "avoids asking only about pay or leave",
        ),
    ),
)

_CLOSING_IDS = {"goals", "candidate_questions"}
_NEGATIVE_EARLY_IDS = {"weakness", "conflict", "mistake", "feedback"}
_GENTLE_OPENERS = ("motivation", "strength", "hire_you", "proud", "learning", "pressure")
_CATEGORIES: dict[str, tuple[str, ...]] = {
    "behavioural": ("teamwork", "conflict", "challenge", "mistake", "feedback", "learning"),
    "self": ("strength", "weakness", "motivation", "hire_you", "proud"),
    "situational": ("pressure", "situational_disagree"),
}


def all_bank() -> dict[str, BankQuestion]:
    """Return every built-in question (HR bank plus extras) by id."""
    return {q.id: q for q in (*HR_QUESTIONS, *EXTRA_QUESTIONS)}


def built_in_ids() -> frozenset[str]:
    """Return the ids that analysis can resolve without stored custom questions."""
    return frozenset(q.id for q in HR_QUESTIONS)


def _pick_core(count: int, rng: random.Random) -> list[BankQuestion]:
    """Pick ``count`` core questions, rotating through categories for variety."""
    bank = all_bank()
    pools = {
        name: [bank[i] for i in ids if i in bank and i not in _CLOSING_IDS]
        for name, ids in _CATEGORIES.items()
    }
    for pool in pools.values():
        rng.shuffle(pool)
    order = list(pools)
    rng.shuffle(order)
    chosen: list[BankQuestion] = []
    while len(chosen) < count and any(pools.values()):
        for name in order:
            if len(chosen) >= count:
                break
            if pools[name]:
                chosen.append(pools[name].pop())
    return chosen


def plan_realistic(
    total: int,
    profile: dict[str, Any] | None,
    llm: LLMClient | None,
    seed: int | None = None,
) -> tuple[BankQuestion, ...]:
    """Return ``total`` distinct questions in a varied, realistic order."""
    if not 1 <= total <= 12:
        raise ValueError("total must be from 1 to 12.")
    rng = random.Random(seed)
    base_intro = HR_QUESTIONS[0]
    intro = BankQuestion(
        base_intro.id, base_intro.topic, rng.choice(INTRO_VARIANTS), base_intro.rubric_hints
    )
    if total == 1:
        return (intro,)

    resume_questions: list[BankQuestion] = []
    if profile:
        wanted = resume_question_count(total, profile)
        if wanted:
            resume_questions = build_resume_questions(profile, wanted, llm)[: total - 1]

    bank = all_bank()
    closing: BankQuestion | None = None
    if total >= 4:
        closing_options = [bank["goals"]]
        if total >= 5:
            closing_options.append(bank["candidate_questions"])
        closing = rng.choice(closing_options)

    core_count = total - 1 - len(resume_questions) - (1 if closing else 0)
    core = _pick_core(max(0, core_count), rng)
    middle = core + resume_questions
    rng.shuffle(middle)
    # Do not open with a negative question; real interviews warm up first.
    if middle and middle[0].id in _NEGATIVE_EARLY_IDS:
        for position, question in enumerate(middle):
            if question.id not in _NEGATIVE_EARLY_IDS:
                middle[0], middle[position] = middle[position], middle[0]
                break
        else:  # every middle question is negative: swap one for a gentler one
            used = {q.id for q in middle} | ({closing.id} if closing else set())
            gentle = [bank[i] for i in _GENTLE_OPENERS if i not in used]
            if gentle:
                middle[0] = rng.choice(gentle)
    plan = [intro, *middle]
    if closing is not None:
        plan.append(closing)
    return tuple(plan[:total])
