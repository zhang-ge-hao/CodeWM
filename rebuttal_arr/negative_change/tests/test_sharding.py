from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import run  # noqa: E402


def _fake_inputs(dataset: str) -> SimpleNamespace:
    task_count = {"humaneval_py": 164, "mbpp_py": 378}[dataset]
    return SimpleNamespace(
        dataset=dataset,
        task_names=tuple(f"{dataset}/{index}" for index in range(task_count)),
        wllm_configs=tuple(
            SimpleNamespace(config_id=f"{index:03d}") for index in range(1, 16)
        ),
    )


class ScoreShardPlanTest(unittest.TestCase):
    def test_global_job_boundaries_are_stable(self) -> None:
        with patch.object(run, "load_dataset_inputs", side_effect=_fake_inputs):
            shards = run.build_score_shards()

        self.assertEqual(len(shards), 285)
        expected = {
            0: ("humaneval_py", "001", 0, 0, 30),
            5: ("humaneval_py", "001", 5, 150, 164),
            89: ("humaneval_py", "015", 5, 150, 164),
            90: ("mbpp_py", "001", 0, 0, 30),
            102: ("mbpp_py", "001", 12, 360, 378),
            284: ("mbpp_py", "015", 12, 360, 378),
        }
        for job_index, values in expected.items():
            with self.subTest(job_index=job_index):
                shard = shards[job_index]
                self.assertEqual(shard.global_index, job_index)
                self.assertEqual(
                    (
                        shard.dataset,
                        shard.config_id,
                        shard.part_index,
                        shard.task_start,
                        shard.task_stop,
                    ),
                    values,
                )
                self.assertLessEqual(shard.task_count, 30)

        self.assertEqual(
            sum(shard.dataset == "humaneval_py" for shard in shards), 90
        )
        self.assertEqual(sum(shard.dataset == "mbpp_py" for shard in shards), 195)

    def test_manifest_describes_the_shard_plan(self) -> None:
        fake_shards = [
            run.ScoreShard(0, "humaneval_py", "001", 0, 0, 30),
            run.ScoreShard(1, "mbpp_py", "001", 0, 0, 30),
        ]
        with (
            patch.object(run, "_input_files", return_value=[]),
            patch.object(run, "_git_commit", return_value=None),
            patch.object(run, "_package_versions", return_value={}),
            patch.object(run, "obfuscator_versions", return_value={}),
            patch.object(run, "build_score_shards", return_value=fake_shards),
        ):
            manifest = run.build_manifest({"datasets": {}}, "tokenizer")

        self.assertEqual(manifest["score_sharding"]["max_tasks_per_job"], 30)
        self.assertEqual(manifest["score_sharding"]["job_count"], 2)
        self.assertEqual(
            manifest["score_sharding"]["dataset_job_counts"],
            {"humaneval_py": 1, "mbpp_py": 1},
        )


class ScoreShardLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.record_root = Path(self.temporary.name)
        self.record_patch = patch.object(run, "RECORD_ROOT", self.record_root)
        self.record_patch.start()
        self.addCleanup(self.record_patch.stop)
        self.inputs = SimpleNamespace(
            dataset="test_dataset",
            task_names=tuple(f"task/{index}" for index in range(31)),
        )
        self.config = SimpleNamespace(
            config_id="001",
            delta=0.5,
            gamma=0.25,
            temperature=1.0,
            ngram_len=5,
        )
        self.shards = [
            run.ScoreShard(0, "test_dataset", "001", 0, 0, 30),
            run.ScoreShard(1, "test_dataset", "001", 1, 30, 31),
        ]

    def _row(self, task: str, shard: run.ScoreShard) -> dict:
        return {
            "dataset": "test_dataset",
            "config": "001",
            "task": task,
            "detector": {
                "delta": 0.5,
                "gamma": 0.25,
                "temperature": 1.0,
                "ngram_len": 5,
            },
            "score_shard": {
                "job_index": shard.global_index,
                "part": shard.part_index,
                "task_start": shard.task_start,
                "task_stop": shard.task_stop,
            },
        }

    def _write(self, shard: run.ScoreShard, rows: list[dict]) -> None:
        shard.output_path.parent.mkdir(parents=True, exist_ok=True)
        shard.output_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def _valid_rows(self, shard: run.ScoreShard) -> list[dict]:
        return [
            self._row(task, shard)
            for task in self.inputs.task_names[shard.task_start : shard.task_stop]
        ]

    def _write_valid_files(self) -> None:
        for shard in self.shards:
            self._write(shard, self._valid_rows(shard))

    def test_complete_shards_are_loaded(self) -> None:
        self._write_valid_files()
        rows = run.load_config_score_rows(
            self.inputs, self.config, self.shards
        )
        self.assertEqual(len(rows), 31)
        self.assertEqual({row["task"] for row in rows}, set(self.inputs.task_names))

    def test_missing_shard_is_rejected(self) -> None:
        self._write(self.shards[0], self._valid_rows(self.shards[0]))
        with self.assertRaisesRegex(FileNotFoundError, "Missing score shard"):
            run.load_config_score_rows(self.inputs, self.config, self.shards)

    def test_extra_shard_is_rejected(self) -> None:
        self._write_valid_files()
        extra = self.shards[0].output_path.parent / "part-002.jsonl"
        extra.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unexpected score shard"):
            run.load_config_score_rows(self.inputs, self.config, self.shards)

    def test_duplicate_task_is_rejected(self) -> None:
        rows = self._valid_rows(self.shards[0])
        rows[1]["task"] = rows[0]["task"]
        self._write(self.shards[0], rows)
        self._write(self.shards[1], self._valid_rows(self.shards[1]))
        with self.assertRaisesRegex(ValueError, "Duplicate tasks"):
            run.load_config_score_rows(self.inputs, self.config, self.shards)

    def test_wrong_task_is_rejected(self) -> None:
        rows = self._valid_rows(self.shards[0])
        rows[0]["task"] = "task/not-in-this-shard"
        self._write(self.shards[0], rows)
        self._write(self.shards[1], self._valid_rows(self.shards[1]))
        with self.assertRaisesRegex(ValueError, "Task coverage mismatch"):
            run.load_config_score_rows(self.inputs, self.config, self.shards)

    def test_wrong_row_count_is_rejected(self) -> None:
        self._write(self.shards[0], self._valid_rows(self.shards[0])[:-1])
        self._write(self.shards[1], self._valid_rows(self.shards[1]))
        with self.assertRaisesRegex(ValueError, "Task coverage mismatch"):
            run.load_config_score_rows(self.inputs, self.config, self.shards)

    def test_detector_config_mismatch_is_rejected(self) -> None:
        rows = self._valid_rows(self.shards[0])
        rows[0]["detector"]["gamma"] = 0.1
        self._write(self.shards[0], rows)
        self._write(self.shards[1], self._valid_rows(self.shards[1]))
        with self.assertRaisesRegex(ValueError, "Detector config mismatch"):
            run.load_config_score_rows(self.inputs, self.config, self.shards)

    def test_shard_metadata_mismatch_is_rejected(self) -> None:
        rows = self._valid_rows(self.shards[0])
        rows[0]["score_shard"]["job_index"] = 99
        self._write(self.shards[0], rows)
        self._write(self.shards[1], self._valid_rows(self.shards[1]))
        with self.assertRaisesRegex(ValueError, "Score shard metadata mismatch"):
            run.load_config_score_rows(self.inputs, self.config, self.shards)


if __name__ == "__main__":
    unittest.main()
