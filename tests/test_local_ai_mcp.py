from __future__ import annotations

import unittest

from testbench.local_ai_test.run import (
    _redact_opaque_text,
    _score,
)


class LocalAIToolUseHarnessTests(unittest.TestCase):
    def test_endpoint_errors_redact_long_opaque_values(self) -> None:
        message = f"Malformed call near {'A' * 5000}"

        redacted = _redact_opaque_text(message)

        self.assertIn("redacted opaque value: 5000 characters", redacted)
        self.assertNotIn("A" * 256, redacted)

    def test_score_accepts_expected_path_tool_arguments(self) -> None:
        case = {
            "expected_tools": ["transcribe_audio_file"],
            "expected_outcomes": ["success"],
            "argument_rules": [{"file_path": {"equals": "sample.mp3"}}],
        }
        result = {
            "invocations": [
                {
                    "name": "transcribe_audio_file",
                    "outcome": "success",
                    "arguments": {"file_path": "sample.mp3"},
                    "error": None,
                }
            ],
            "final_text": "The file was transcribed.",
        }

        score = _score(case, result)

        self.assertTrue(score["passed"], score["failures"])


if __name__ == "__main__":
    unittest.main()
