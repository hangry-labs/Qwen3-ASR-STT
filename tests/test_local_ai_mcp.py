from __future__ import annotations

import unittest

from testbench.local_ai_test.run import (
    _redact_opaque_text,
    _replace_audio_url,
    _score,
)


class LocalAIToolUseHarnessTests(unittest.TestCase):
    def test_endpoint_errors_redact_long_opaque_values(self) -> None:
        message = f"Malformed call near {'A' * 5000}"

        redacted = _redact_opaque_text(message)

        self.assertIn("redacted opaque value: 5000 characters", redacted)
        self.assertNotIn("A" * 256, redacted)

    def test_score_accepts_expected_file_location_arguments(self) -> None:
        case = {
            "expected_tools": ["transcribe_audio_file"],
            "expected_outcomes": ["success"],
            "argument_rules": [{"file_location": {"equals": "sample.mp3"}}],
        }
        result = {
            "invocations": [
                {
                    "name": "transcribe_audio_file",
                    "outcome": "success",
                    "arguments": {"file_location": "sample.mp3"},
                    "error": None,
                }
            ],
            "final_text": "The file was transcribed.",
        }

        score = _score(case, result)

        self.assertTrue(score["passed"], score["failures"])

    def test_audio_url_placeholder_is_replaced_in_prompt_and_rules(self) -> None:
        case = {
            "prompt": "Transcribe {audio_url}",
            "argument_rules": [
                {"file_location": {"equals": "{audio_url}"}},
            ],
        }

        replaced = _replace_audio_url(case, "http://tts:8000/generated.wav")

        self.assertEqual(
            replaced["prompt"],
            "Transcribe http://tts:8000/generated.wav",
        )
        self.assertEqual(
            replaced["argument_rules"][0]["file_location"]["equals"],
            "http://tts:8000/generated.wav",
        )


if __name__ == "__main__":
    unittest.main()
