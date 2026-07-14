from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_mini_experiment as experiment  # noqa: E402
from mini_qwen_model import QwenHFTextModelConfig  # noqa: E402


class AddedCodeExtractionTest(unittest.TestCase):
    def test_extracts_only_added_hunk_lines_in_diff_order(self) -> None:
        patch = """diff --git a/a.py b/a.py
index 111..222 100644
--- a/a.py
+++ b/a.py
@@ -1,2 +1,4 @@
 keep
-old
+new
+
+++counter
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -3 +3 @@
-x = 1
+x = 2
"""
        self.assertEqual(experiment.extract_added_code(patch), "new\n\n++counter\nx = 2\n")

    def test_empty_or_header_only_diff_has_no_added_code(self) -> None:
        patch = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
"""
        self.assertEqual(experiment.extract_added_code(patch), "")

    def test_preserves_added_line_without_terminal_newline(self) -> None:
        patch = "@@ -0,0 +1 @@\n+value"
        self.assertEqual(experiment.extract_added_code(patch), "value\n")

    def test_detector_receives_all_and_only_added_lines(self) -> None:
        class RecordingTokenizer:
            text = None

            def __call__(self, text, *, add_special_tokens):
                self.text = text
                self.asserted_without_special_tokens = not add_special_tokens
                return {"input_ids": [10, 11, 12, 13, 14, 15]}

        class RecordingDetector:
            tokenized_text = None

            def detect(self, *, tokenized_text, tokenized_prefix, **kwargs):
                self.tokenized_text = tokenized_text.tolist()
                self.tokenized_prefix = tokenized_prefix.tolist()
                return {
                    "num_tokens_scored": 2,
                    "num_green_tokens": 1,
                    "green_fraction": 0.5,
                    "z_score": 0.0,
                    "p_value": 0.5,
                    "prediction": False,
                }

        patch = "--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n-old\n+new\n+next\n"
        tokenizer = RecordingTokenizer()
        detector = RecordingDetector()
        added, result = experiment.detect_added_code(
            patch,
            detector,
            tokenizer,
            SimpleNamespace(z_threshold=4.0),
        )
        self.assertEqual(added, "new\nnext\n")
        self.assertEqual(tokenizer.text, added)
        self.assertTrue(tokenizer.asserted_without_special_tokens)
        self.assertEqual(detector.tokenized_text, [10, 11, 12, 13, 14, 15])
        self.assertEqual(detector.tokenized_prefix, [])
        self.assertEqual(result["scope"], "all_patch_added_lines")


class DeadlineAgentTest(unittest.TestCase):
    def test_deadline_submits_current_diff(self) -> None:
        from minisweagent.exceptions import Submitted

        class FakeModel:
            def format_message(self, **kwargs):
                return kwargs

            def get_template_vars(self):
                return {}

            def serialize(self):
                return {}

        class FakeEnvironment:
            def __init__(self):
                self.commands = []

            def get_template_vars(self):
                return {}

            def execute(self, action):
                self.commands.append(action["command"])
                raise Submitted(
                    {
                        "role": "exit",
                        "content": "diff --git a/x.py b/x.py\n",
                        "extra": {
                            "exit_status": "Submitted",
                            "submission": "diff --git a/x.py b/x.py\n",
                        },
                    }
                )

            def serialize(self):
                return {}

        env = FakeEnvironment()
        agent = experiment.create_deadline_agent(
            FakeModel(),
            env,
            system_template="system",
            instance_template="task",
            step_limit=250,
            cost_limit=0,
            output_path=None,
            wall_time_limit_seconds=1,
        )
        with patch.object(experiment.time, "monotonic", side_effect=[0.0, 2.0]):
            result = agent.run("problem")

        self.assertEqual(env.commands, [experiment.FORCED_SUBMIT_COMMAND])
        self.assertIn("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", env.commands[0])
        self.assertIn("test -s patch.txt", env.commands[0])
        self.assertIn("git diff -- .", env.commands[0])
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(result["submission"], "diff --git a/x.py b/x.py\n")
        self.assertTrue(agent.deadline_forced_submission)


class GenerationTimeLimitTest(unittest.TestCase):
    def test_generation_time_limit_is_five_minutes(self) -> None:
        config = QwenHFTextModelConfig()
        self.assertEqual(config.max_new_tokens, 32_768)
        self.assertEqual(config.max_generation_seconds, 300.0)


if __name__ == "__main__":
    unittest.main()
