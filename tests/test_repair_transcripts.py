import json

from scripts.tools.repair_transcripts import main, repair_directory, repair_session


def test_repair_session_rebuilds_only_whitespace_merged_text():
    original = {
        "turns": [
            {
                "answer": {
                    "text": "We saidHello, today.",
                    "words": [{"text": "We"}, {"text": " said"}, {"text": "Hello,"}, {"text": " today."}],
                }
            },
            {
                "answer": {
                    "text": "Keep this.",
                    "words": [{"text": "Keep"}, {"text": " this."}],
                }
            },
            {
                "answer": {
                    "text": "I need a-",
                    "words": [
                        {"text": "I"},
                        {"text": " need"},
                        {"text": " a"},
                        {"text": "-"},
                    ],
                }
            },
        ]
    }

    repaired, changed = repair_session(original)

    assert changed == 1
    assert repaired["turns"][0]["answer"]["text"] == "We said Hello, today."
    assert repaired["turns"][1]["answer"]["text"] == "Keep this."
    assert repaired["turns"][2]["answer"]["text"] == "I need a-"


def test_repair_directory_dry_run_and_write_never_change_originals(tmp_path, capsys):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    session_path = sessions / "session.json"
    original = {
        "turns": [
            {
                "answer": {
                    "text": "butI went.",
                    "words": [{"text": "but"}, {"text": "I"}, {"text": " went."}],
                }
            }
        ]
    }
    session_path.write_text(json.dumps(original), encoding="utf-8")
    output = tmp_path / "repaired"

    assert repair_directory(sessions, output) == (1, 1)
    assert not output.exists()
    assert json.loads(session_path.read_text(encoding="utf-8")) == original
    assert main(
        ["--sessions-dir", str(sessions), "--output-dir", str(output), "--write"]
    ) == 0
    assert json.loads(session_path.read_text(encoding="utf-8")) == original
    assert json.loads((output / "session.json").read_text(encoding="utf-8"))[
        "turns"
    ][0]["answer"]["text"] == "but I went."
    assert "Changed turns: 1" in capsys.readouterr().out
