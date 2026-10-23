from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_mini_vllm_experiment as experiment


class OfficialWorkflowTest(unittest.TestCase):
    def test_uses_tool_call_config_with_30_minute_runner_deadline(self) -> None:
        config = experiment.load_official_config()
        self.assertEqual(config["agent"]["step_limit"], 250)
        self.assertNotIn("wall_time_limit_seconds", config["agent"])
        self.assertEqual(experiment.AGENT_WALL_TIME_LIMIT_SECONDS, 1800.0)
        self.assertNotIn("mswea_bash_command", config["agent"]["system_template"])
        self.assertIn("bash tool call", config["agent"]["instance_template"])

    def test_modal_sandbox_has_cleanup_grace_after_agent_deadline(self) -> None:
        config = experiment.load_official_config()
        self.assertEqual(config["environment"]["deployment_timeout"], 1800.0)

        experiment.ensure_modal_sandbox_outlives_agent(config, 1800.0)

        self.assertEqual(config["environment"]["deployment_timeout"], 2100.0)
        self.assertEqual(
            experiment.MODAL_SANDBOX_CLEANUP_GRACE_SECONDS, 300.0
        )

    def test_modal_startup_timeout_is_retried_until_success(self) -> None:
        from modal.exception import SandboxTimeoutError

        expected_environment = object()
        with patch(
            "minisweagent.run.benchmarks.swebench.get_sb_environment",
            side_effect=[
                SandboxTimeoutError(),
                SandboxTimeoutError(),
                expected_environment,
            ],
        ) as get_environment:
            environment = experiment.get_sb_environment_with_retries(
                {},
                {"instance_id": "test__case-1"},
                attempts=5,
                index=1,
                total=1,
            )

        self.assertIs(environment, expected_environment)
        self.assertEqual(get_environment.call_count, 3)

    def test_modal_startup_timeout_raises_after_all_attempts(self) -> None:
        from modal.exception import SandboxTimeoutError

        with patch(
            "minisweagent.run.benchmarks.swebench.get_sb_environment",
            side_effect=SandboxTimeoutError(),
        ) as get_environment:
            with self.assertRaises(SandboxTimeoutError):
                experiment.get_sb_environment_with_retries(
                    {},
                    {"instance_id": "test__case-1"},
                    attempts=5,
                    index=1,
                    total=1,
                )

        self.assertEqual(get_environment.call_count, 5)

    def test_modal_environment_stop_explicitly_terminates_sandbox(self) -> None:
        environment = SimpleNamespace(
            deployment=SimpleNamespace(
                _sandbox=SimpleNamespace(object_id="sb-test-123")
            ),
            stop=Mock(),
        )
        sandbox = Mock()
        with patch("modal.Sandbox.from_id", return_value=sandbox) as from_id:
            experiment.stop_sb_environment(environment)

        environment.stop.assert_called_once_with()
        from_id.assert_called_once_with("sb-test-123")
        sandbox.terminate.assert_called_once_with(wait=False)

    def test_wllm_parameters_are_request_scoped(self) -> None:
        args = SimpleNamespace(
            generation_seed=7,
            served_model_name="Qwen3.5-35B-A3B-GPTQ-Int4",
            vllm_base_url="http://127.0.0.1:8000",
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            max_new_tokens=32768,
            watermarking="wllm",
            gamma=0.5,
            delta=4.0,
            watermark_key=15485863,
        )
        model = experiment.make_model(
            args, experiment.load_official_config(), vocab_size=248077
        )
        kwargs = model.config.model_kwargs
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["extra_body"]["top_k"], 20)
        self.assertEqual(
            kwargs["extra_body"]["vllm_xargs"],
            {
                "wllm_enabled": 1,
                "wllm_gamma": 0.5,
                "wllm_delta": 4.0,
                "wllm_ngram_len": 5,
                "wllm_hash_key": 15485863,
                "wllm_vocab_size": 248077,
            },
        )


if __name__ == "__main__":
    unittest.main()
