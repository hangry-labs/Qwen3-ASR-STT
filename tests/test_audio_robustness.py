import json
from pathlib import Path

from testbench import run_audio_robustness


def _manifest() -> dict:
    path = Path("testbench/robustness_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_expands_auto_and_forced_language_cases() -> None:
    cases = run_audio_robustness._expand_cases(_manifest())

    assert len(cases) == 44
    assert sum(case["expected_kind"] == "empty" for case in cases) == 30
    assert sum(case["expected_kind"] == "source" for case in cases) == 14
    assert {case["id"] for case in cases} >= {
        "silence_5s__auto",
        "silence_5s__chinese",
        "english_weak_40db__english",
    }


def test_empty_fixture_rejects_lexical_hallucination() -> None:
    case = {"expected_kind": "empty"}

    assert run_audio_robustness._evaluate(case, "") == (True, 100.0)
    assert run_audio_robustness._evaluate(case, "...") == (True, 100.0)
    assert run_audio_robustness._evaluate(case, "嗯。") == (False, 0.0)


def test_speech_fixture_uses_minimum_meaning_score() -> None:
    case = {
        "expected_kind": "source",
        "expected_text": "The kettle is ready.",
        "minimum_score": 90.0,
    }

    passed, score = run_audio_robustness._evaluate(case, "The kettle is ready!")
    failed, failed_score = run_audio_robustness._evaluate(case, "The.")

    assert passed is True
    assert score == 100.0
    assert failed is False
    assert failed_score < 90.0


def test_speech_noise_command_keeps_source_duration(tmp_path: Path) -> None:
    generator = {
        "kind": "speech_noise",
        "source": "english",
        "speech_volume": 0.03,
        "noise_color": "white",
        "noise_amplitude": 0.02,
        "seed": 165,
    }
    sources = {"english": {"audio": "assets/english/random/01.mp3"}}

    command = run_audio_robustness._ffmpeg_command(generator, sources, Path("testbench"), tmp_path / "out.wav")

    assert any("amix=inputs=2:duration=first:normalize=0[out]" in argument for argument in command)
    assert command[-7:] == ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp_path / "out.wav")]
