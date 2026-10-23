from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_python_minifier_postprocess as postprocess


class PatchPathTest(unittest.TestCase):
    def test_finds_modified_and_added_python_post_images(self) -> None:
        patch = """diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1 +1 @@
-x = 1
+x = 2
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
diff --git a/pkg/new.py b/pkg/new.py
new file mode 100644
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1 @@
+value = 1
"""
        self.assertEqual(
            postprocess.python_paths_from_patch(patch),
            ("pkg/a.py", "pkg/new.py"),
        )
        self.assertEqual(
            postprocess.all_paths_from_patch(patch),
            ("pkg/a.py", "README.md", "pkg/new.py"),
        )

    def test_ignores_deleted_python_file(self) -> None:
        patch = """diff --git a/pkg/a.py b/pkg/a.py
deleted file mode 100644
--- a/pkg/a.py
+++ /dev/null
@@ -1 +0,0 @@
-x = 1
"""
        self.assertEqual(postprocess.python_paths_from_patch(patch), ())
        self.assertEqual(postprocess.all_paths_from_patch(patch), ("pkg/a.py",))

    def test_extracts_only_positive_post_image_hunk_lines(self) -> None:
        patch = """@@ -10,2 +10,3 @@
-old
+new
@@ -30,2 +31,0 @@
-deleted
"""
        self.assertEqual(postprocess.changed_post_image_lines(patch), {10, 11, 12})


class FunctionSelectionTest(unittest.TestCase):
    def test_selects_only_method_containing_changed_line(self) -> None:
        source = """CONSTANT = 1

class Example:
    def untouched(self):
        return 1

    def changed(self, value):
        return value + 1

def sibling():
    return 3
"""
        spans = postprocess.modified_function_spans(source, {8})
        self.assertEqual(
            spans,
            [postprocess.FunctionSpan(7, 8, "Example.changed")],
        )

    def test_nested_edit_selects_inner_function_until_outer_is_also_changed(self) -> None:
        source = """def outer(value):
    adjusted = value + 1
    def inner():
        return adjusted
    return inner()
"""
        self.assertEqual(
            postprocess.modified_function_spans(source, {4}),
            [postprocess.FunctionSpan(3, 4, "outer.inner")],
        )
        self.assertEqual(
            postprocess.modified_function_spans(source, {2, 4}),
            [postprocess.FunctionSpan(1, 5, "outer")],
        )


class AddedLineMetricTest(unittest.TestCase):
    def test_uses_multiset_line_precision_recall_and_f1(self) -> None:
        original = """diff --git a/a.py b/a.py
@@ -0,0 +1,3 @@
+same
+same
+old-only
"""
        final = """diff --git a/a.py b/a.py
@@ -0,0 +1,2 @@
+same
+new-only
"""
        metrics = postprocess.added_line_metrics(original, final)
        self.assertEqual(metrics["original_added_lines"], 3)
        self.assertEqual(metrics["final_added_lines"], 2)
        self.assertEqual(metrics["overlapping_added_lines"], 1)
        self.assertAlmostEqual(metrics["added_line_precision"], 0.5)
        self.assertAlmostEqual(metrics["added_line_recall"], 1 / 3)
        self.assertAlmostEqual(metrics["added_line_f1"], 0.4)

    def test_empty_added_code_matches_itself(self) -> None:
        metrics = postprocess.added_line_metrics("", "")
        self.assertEqual(metrics["added_line_precision"], 1.0)
        self.assertEqual(metrics["added_line_recall"], 1.0)
        self.assertEqual(metrics["added_line_f1"], 1.0)


class RepositoryTransformTest(unittest.TestCase):
    def test_applies_patch_from_checkout_and_rebuilds_diff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=source, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=source,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=source, check=True
            )
            (source / "module.py").write_text("def value():\n    return 1\n")
            subprocess.run(["git", "add", "module.py"], cwd=source, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=source, check=True)
            base_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            (source / "module.py").write_text("def value():\n    return 2\n")
            original_patch = subprocess.run(
                ["git", "diff", "--full-index"],
                cwd=source,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            mirror = root / "mirror.git"
            subprocess.run(
                ["git", "clone", "-q", "--bare", str(source), str(mirror)],
                check=True,
            )
            fake_minifier = root / "pyminify"
            fake_minifier.touch()
            real_run_command = postprocess.run_command

            def run_with_fake_minifier(command, **kwargs):
                if command[0] == str(fake_minifier):
                    transformed = (Path(kwargs["cwd"]) / command[-1]).read_text().replace(
                        "return 2", "return(2)"
                    )
                    return subprocess.CompletedProcess(
                        command, 0, stdout=transformed, stderr=""
                    )
                return real_run_command(command, **kwargs)

            with patch.object(
                postprocess, "ensure_repo_mirror", return_value=mirror
            ), patch.object(
                postprocess, "run_command", side_effect=run_with_fake_minifier
            ):
                result = postprocess.minify_staged_python_patch(
                    repo="owner/repo",
                    base_commit=base_commit,
                    original_patch=original_patch,
                    cache_root=root / "cache",
                    pyminify=fake_minifier,
                    work_root=root / "work",
                )
            self.assertEqual(result.status, "transformed", result.error)
            self.assertEqual(result.transformed_python_paths, ("module.py",))
            self.assertEqual(
                result.transformed_python_functions,
                ("module.py:value@1-2",),
            )
            self.assertEqual(result.obfuscator_invocations, 1)
            self.assertGreaterEqual(result.obfuscator_seconds, 0.0)
            self.assertIn("+    return(2)", result.final_patch)


class SelectionTest(unittest.TestCase):
    def test_explicit_ids_preserve_original_order_and_indices(self) -> None:
        selection = {"instance_ids": ["a", "b", "c"]}
        dataset = {
            value: {"instance_id": value, "repo": "o/r", "base_commit": value}
            for value in selection["instance_ids"]
        }
        args = SimpleNamespace(instance_id=["c", "a"], limit=None)
        ids, cases, indices = postprocess.select_cases(selection, dataset, args)
        self.assertEqual(ids, ["a", "c"])
        self.assertEqual([case["instance_id"] for case in cases], ["a", "c"])
        self.assertEqual(indices, {"a": 1, "b": 2, "c": 3})


class SummaryTest(unittest.TestCase):
    def test_reports_paired_quality_and_detection(self) -> None:
        rows = {
            "a": {
                "patch_identical": False,
                "obfuscation": {"status": "transformed"},
                "pre_detection": {"z_score": 2.0, "invalid": False},
                "post_detection": {"z_score": 0.0, "invalid": False},
            },
            "b": {
                "patch_identical": True,
                "obfuscation": {"status": "empty_patch"},
                # These deliberately large valid scores must not leak into
                # post-obfuscation metrics because no transform occurred.
                "pre_detection": {"z_score": 9.0, "invalid": False},
                "post_detection": {"z_score": 9.0, "invalid": False},
            },
        }
        summary = postprocess.make_summary(
            rows=rows,
            selected_ids=["a", "b"],
            original_report={"resolved_ids": ["a"]},
            post_report={"resolved_ids": ["a", "b"]},
            settings={"watermarking": "wllm", "delta": 2.0, "gamma": 0.5},
        )
        self.assertEqual(summary["transformed_cases"], 1)
        self.assertEqual(summary["original_solve_rate"], 0.5)
        self.assertEqual(summary["post_solve_rate"], 1.0)
        self.assertEqual(summary["retained_resolved_rate"], 1.0)
        self.assertEqual(summary["mean_paired_z_change"], -2.0)
        self.assertEqual(summary["paired_detection_cases"], 1)
        self.assertEqual(summary["mean_pre_z_score"], 2.0)
        self.assertEqual(summary["mean_post_z_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
