"""Offline tests for content retries, stability, and student/model agreement."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import analyze_session as analyze_cli
from scripts.content_stability import stability_rows
from scripts.eval_content import evaluate_ratings, spearman_correlation
from scripts.make_report import main as make_report_main
from src.speaksure.analysis.content import analyze_content
from src.speaksure.analysis.session_analysis import analyze_session
from src.speaksure.llm_client import LLMError
from src.speaksure.question_bank import HR_QUESTIONS
from src.speaksure.session_log import InterviewSession
from src.speaksure.speech_to_text import TranscriptionResult


QUESTION = HR_QUESTIONS[0]
ANSWER = (
    "I am a software developer and I built a useful scheduling tool "
    "for my university team."
)


def _valid_response(score: int = 4) -> str:
    """Build a correctly shaped response for the introduction rubric."""
    return json.dumps(
        {
            "relevance": score,
            "specificity": score,
            "clarity": score,
            "coverage": [
                {"hint": hint, "covered": False, "evidence": ""}
                for hint in QUESTION.rubric_hints
            ],
            "star": None,
            "strengths": [],
            "improvements": [],
            "reason": "A clear test answer.",
        }
    )


def _analyze_with(fake_llm, **kwargs):
    """Call the real content scorer with one sufficiently long synthetic answer."""
    session = InterviewSession.new("hr", "standard", 1)
    turn = session.add_turn(
        QUESTION.text,
        answer=TranscriptionResult(ANSWER, (), 2.0, "en", 1.0, "test"),
    )
    return analyze_content(
        [turn],
        QUESTION.id,
        {QUESTION.id: QUESTION},
        "engineer",
        fake_llm,
        asr_mean_probability=0.9,
        asr_reliability="high",
        include_attempts=True,
        **kwargs,
    )


class _SequenceLLM:
    """Return successive JSON outputs or raise successive safe LLM errors."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, *_args, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome, model="fake-model")


def test_transient_retry_succeeds_on_second_attempt_and_records_attempt_count(
    monkeypatch,
):
    llm = _SequenceLLM([LLMError("rate limited"), _valid_response()])
    delays = []
    monkeypatch.setattr(
        "src.speaksure.analysis.content.random.uniform", lambda low, high: 0.1
    )

    result = _analyze_with(llm, sleep=delays.append)

    assert result["rated"] is True
    assert result["attempts"] == 2
    assert llm.calls == 2
    assert delays == [pytest.approx(2.1)]


def test_rate_limit_retries_at_most_three_times_and_keeps_failure_category(
    monkeypatch,
):
    llm = _SequenceLLM([LLMError("rate limited") for _ in range(4)])
    delays = []
    monkeypatch.setattr(
        "src.speaksure.analysis.content.random.uniform", lambda low, high: 0.0
    )

    result = _analyze_with(llm, sleep=delays.append)

    assert result == {
        "analyzer_version": 2,
        "rated": False,
        "error": "llm_error",
        "attempts": 4,
    }
    assert llm.calls == 4
    assert delays == [2.0, 4.0, 8.0]


def test_invalid_output_retries_only_once_and_records_attempts():
    llm = _SequenceLLM(["not json", "still not json"])

    result = _analyze_with(llm, sleep=lambda _delay: pytest.fail("unexpected sleep"))

    assert result["rated"] is False
    assert result["error"] == "invalid_output"
    assert result["attempts"] == 2
    assert llm.calls == 2


def test_session_question_group_records_the_llm_attempt_count():
    session = InterviewSession.new("hr", "standard", 1)
    session.add_turn(
        QUESTION.text,
        answer=TranscriptionResult(ANSWER, (), 2.0, "en", 1.0, "test"),
        analysis={"asked": {"main_question_id": QUESTION.id, "is_follow_up": False}},
    )

    analyzed = analyze_session(
        session,
        llm=_SequenceLLM([_valid_response()]),
    )

    assert analyzed.analysis["content_by_question"][0]["attempts"] == 1


def test_repeat_three_saves_runs_and_selects_median_score_run(
    tmp_path, monkeypatch, capsys
):
    session = InterviewSession.new("hr", "standard", 1)
    session.add_turn(
        QUESTION.text,
        answer=TranscriptionResult(ANSWER, (), 2.0, "en", 1.0, "test"),
        analysis={"asked": {"main_question_id": QUESTION.id, "is_follow_up": False}},
    )
    source_path = session.save_to(tmp_path / "source.json")
    output_path = tmp_path / "analyzed.json"
    run_scores = iter((70, 78, 74))

    monkeypatch.setattr(
        analyze_cli.GroqAnalysisLLMConfig,
        "from_env",
        classmethod(
            lambda cls: SimpleNamespace(api_key="fake-key", model="fake-model")
        ),
    )
    monkeypatch.setattr(analyze_cli, "build_analysis_llm_client", lambda _config: object())

    def fake_analyze(source, **_kwargs):
        result = copy.deepcopy(source)
        score = next(run_scores)
        result.analysis["content_by_question"] = [
            {
                "main_question_id": QUESTION.id,
                "turn_indexes": [1],
                "label_source": "recorded",
                "rated": True,
                "relevance": 4,
                "specificity": 4,
                "clarity": 4,
                "content_score": score,
                "content_components": {"relevance": 0.75},
                "coverage_ratio": 0.0,
                "strengths": [],
                "improvements": [],
            }
        ]
        return result

    monkeypatch.setattr(analyze_cli, "analyze_session", fake_analyze)
    monkeypatch.setattr(analyze_cli, "sleep", lambda _seconds: None)

    result = analyze_cli.main(
        [
            str(source_path),
            "--content",
            "--repeat",
            "3",
            "--pause-seconds",
            "0",
            "--out",
            str(output_path),
        ]
    )

    assert result == 0
    analyzed = InterviewSession.load(output_path)
    assert [
        run["content_by_question"][0]["content_score"]
        for run in analyzed.analysis["content_runs"]
    ] == [70, 78, 74]
    median_group = analyzed.analysis["content_by_question"][0]
    assert median_group["content_score"] == 74
    assert median_group["selected_run_index"] == 3
    score_spread = next(
        row
        for row in stability_rows(analyzed.to_dict())
        if row["metric"] == "content_score"
    )
    assert score_spread["band_disagreements"] == 0
    assert "Analysis saved:" in capsys.readouterr().out


def test_cli_refuses_existing_analyzed_output_without_force(tmp_path, capsys):
    session = InterviewSession.new("hr", "standard", 1)
    source_path = session.save_to(tmp_path / "source.json")
    output_path = tmp_path / "existing.json"
    output_path.write_text("keep", encoding="utf-8")

    result = analyze_cli.main(
        [str(source_path), "--out", str(output_path)]
    )

    assert result == 1
    assert output_path.read_text(encoding="utf-8") == "keep"
    assert "already exists" in capsys.readouterr().err

    assert analyze_cli.main(
        [str(source_path), "--out", str(output_path), "--force"]
    ) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_stability_statistics_match_hand_computed_spread():
    # Scores 20, 40, 70 have mean 43.333, population SD 20.548, and range 50.
    analyzed = {
        "analysis": {
            "content_runs": [
                {
                    "content_by_question": [
                        {
                            "main_question_id": "intro",
                            "turn_indexes": [1],
                            "rated": True,
                            "content_score": score,
                            "content_components": {"relevance": score / 100},
                        }
                    ]
                }
                for score in (20, 40, 70)
            ]
        }
    }

    rows = stability_rows(analyzed)
    score_row = next(row for row in rows if row["metric"] == "content_score")
    component_row = next(row for row in rows if row["metric"] == "relevance")

    assert score_row["n"] == 3
    assert score_row["mean"] == pytest.approx(130 / 3)
    assert score_row["standard_deviation"] == pytest.approx(20.5480466766)
    assert score_row["range"] == 50
    assert score_row["band_disagreements"] == 2
    assert component_row["mean"] == pytest.approx(130 / 300)


def test_spearman_uses_average_ranks_for_ties():
    # Average ranks are [1, 2.5, 2.5, 4] and [4, 1.5, 1.5, 3], rho=-1/3.
    assert spearman_correlation(
        [1, 2, 2, 4],
        [4, 1, 1, 2],
    ) == pytest.approx(-1 / 3)


def test_e3_counts_band_agreement_and_lists_unrated_reason(tmp_path):
    analyzed_path = tmp_path / "session-a.analyzed.json"
    analyzed_path.write_text(
        json.dumps(
            {
                "session_id": "session-a",
                "analysis": {
                    "content_by_question": [
                        {
                            "main_question_id": "intro",
                            "rated": True,
                            "content_score": 70,
                            "relevance": 4,
                            "specificity": 3,
                            "clarity": 5,
                        },
                        {
                            "main_question_id": "teamwork",
                            "rated": False,
                            "reason": "too_short_or_empty",
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    ratings = [
        {
            "session": "session-a",
            "question_id": "intro",
            "relevance": "4",
            "specificity": "3",
            "clarity": "5",
        },
        {
            "session": "session-a",
            "question_id": "teamwork",
            "relevance": "2",
            "specificity": "2",
            "clarity": "3",
        },
    ]

    result = evaluate_ratings(ratings, [analyzed_path])

    assert result["n"] == 1
    assert result["band_agreements"] == 1
    assert result["unrated"] == [
        {
            "session": "session-a",
            "question_id": "teamwork",
            "reason": "too_short_or_empty",
        }
    ]


def test_eval_reports_component_correlations_and_student_model_difference(tmp_path):
    analyzed_path = tmp_path / "session-b.json"
    groups = []
    ratings = []
    for index, values in enumerate(
        [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 1), (5, 1, 2)]
    ):
        question_id = f"q{index}"
        groups.append(
            {
                "main_question_id": question_id,
                "rated": True,
                "content_score": 50,
                "relevance": values[0],
                "specificity": values[1],
                "clarity": values[2],
            }
        )
        ratings.append(
            {
                "session": "session-b",
                "question_id": question_id,
                "relevance": str(values[0]),
                "specificity": str(values[1]),
                "clarity": str(values[2]),
            }
        )
    analyzed_path.write_text(
        json.dumps(
            {
                "session_id": "session-b",
                "analysis": {"content_by_question": groups},
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_ratings(ratings, [analyzed_path])

    assert result["n"] == 5
    assert result["overall_spearman"] is None
    assert result["component_correlations"]["relevance"]["n"] == 5
    assert result["mean_absolute_difference"] == pytest.approx(40 / 3)
    assert result["band_agreements"] == 3


@pytest.mark.parametrize(
    "content_runs",
    [
        None,
        [{"content_by_question": []}],
    ],
)
def test_report_cli_renders_json_with_or_without_content_runs(
    tmp_path, content_runs, capsys
):
    analyzed = InterviewSession.new("hr", "standard", 1).to_dict()
    if content_runs is not None:
        analyzed["analysis"]["content_runs"] = content_runs
    source_path = tmp_path / (
        "with-runs.json" if content_runs is not None else "without-runs.json"
    )
    source_path.write_text(json.dumps(analyzed), encoding="utf-8")
    output_path = tmp_path / f"{source_path.stem}.html"

    result = make_report_main(
        [str(source_path), "--out", str(output_path), "--generated-at", "fixed"]
    )

    assert result == 0
    assert "<!doctype html>" in output_path.read_text(encoding="utf-8")
    assert str(output_path) in capsys.readouterr().out
