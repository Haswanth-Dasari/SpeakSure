"""Run a typed-text HR mock interview in the terminal."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.interviewer import InterviewPlan, Interviewer  # noqa: E402
from speaksure.llm_client import GroqLLMConfig, build_llm_client  # noqa: E402
from speaksure.question_bank import select_questions  # noqa: E402
from speaksure.session_log import InterviewSession  # noqa: E402
from speaksure.speech_to_text import TranscriptionResult  # noqa: E402


def main() -> int:
    """Run and optionally save a typed-text mock interview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--follow-ups", type=int, default=1)
    parser.add_argument("--role", default="software engineering fresher")
    parser.add_argument("--name", default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    try:
        questions = select_questions(args.questions, seed=args.seed)
        plan = InterviewPlan(
            questions=questions,
            role=args.role,
            candidate_name=args.name,
            max_followups_per_question=args.follow_ups,
        )
        config = GroqLLMConfig.from_env()
        interviewer = Interviewer(build_llm_client(config), plan)
    except ValueError as error:
        parser.error(str(error))

    session = InterviewSession.new(
        mode="hr",
        difficulty="standard",
        planned_questions=len(questions),
        metadata={"llm_model": config.model, "seed": args.seed},
    )
    opening = interviewer.opening()
    print(f"Interviewer [{opening.kind}]: {opening.utterance}")

    active_question = questions[0].text
    followups = 0
    repeats = 0
    fallbacks = 0
    latencies: list[float] = []
    try:
        while not interviewer.is_finished:
            answer = input("You: ")
            turn = interviewer.next_turn(answer)
            print(f"Interviewer [{turn.kind}]: {turn.utterance}")

            session.add_turn(
                active_question,
                answer=TranscriptionResult(
                    text=answer,
                    words=(),
                    duration_seconds=0.0,
                    language="en",
                    language_probability=1.0,
                    device="typed",
                ),
                analysis={
                    "interviewer": {
                        "rationale": turn.rationale,
                        "kind": turn.kind,
                        "used_fallback": turn.used_fallback,
                        "llm_latency_seconds": turn.llm_latency_seconds,
                    }
                },
            )
            if turn.kind == "follow_up":
                followups += 1
                active_question = turn.utterance
            elif turn.kind == "repeat":
                repeats += 1
            elif turn.kind in {"next_question", "closing"}:
                if turn.kind == "next_question":
                    active_question = turn.utterance
            if turn.used_fallback:
                fallbacks += 1
            if turn.llm_latency_seconds is not None:
                latencies.append(turn.llm_latency_seconds)
    except KeyboardInterrupt:
        print("\nInterview ended early.")

    if not args.no_save:
        saved_path = session.save("data/sessions")
        print(f"Saved interview: {saved_path}")

    average_latency = sum(latencies) / len(latencies) if latencies else 0.0
    print(
        "Summary: "
        f"{len(session.turns)} exchanges, {followups} follow-ups, "
        f"{repeats} repeats, {fallbacks} fallbacks, "
        f"{average_latency:.3f}s average LLM latency."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
