"""Full reference-corpus differential test.

This module is deliberately marked ``long``.  It is not exercised by the
short development loop: running it executes all 164 HumanEval and 378 MBPP
reference programs once as a baseline, then tests three independent twenty-step
random walks from each original program.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing

import pytest

from rw_obfuscator import RandomWalkObfuscator
from rw_obfuscator.corpus import CorpusCase, execute_case, reference_cases


WORKERS = 10
WALKS_PER_CASE = 3
STEPS_PER_WALK = 20
CASE_TIMEOUT_SECONDS = 10.0
CASE_MEMORY_MB = 1024


def _case_seed(case: CorpusCase, variant_index: int) -> int:
    identity = (
        f"{case.dataset}\0{case.task_name}\0{variant_index}".encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")


def _run_reference_case(case: CorpusCase) -> dict[str, object]:
    """Worker entry point; it must stay at module scope for multiprocessing."""

    baseline = execute_case(
        case.source,
        case.test,
        timeout=CASE_TIMEOUT_SECONDS,
        memory_mb=CASE_MEMORY_MB,
    )
    common: dict[str, object] = {
        "dataset": case.dataset,
        "task": case.task_name,
        "baseline": baseline.to_dict(),
    }
    if not baseline.passed:
        return {**common, "status": "baseline_failed"}

    variants: list[dict[str, object]] = []
    output_digests: set[str] = set()
    for variant_index in range(WALKS_PER_CASE):
        seed = _case_seed(case, variant_index)
        variant: dict[str, object] = {
            "variant": variant_index,
            "seed": seed,
        }
        try:
            # Each trajectory starts from the original program with a fresh
            # engine. These are three independent ten-step walks, not one
            # thirty-step walk split into three checkpoints.
            walked = RandomWalkObfuscator(case.source, seed=seed).walk(
                case.source,
                STEPS_PER_WALK,
            )
            digest = hashlib.sha256(walked.source.encode("utf-8")).hexdigest()
            output_digests.add(digest)
            obfuscated = execute_case(
                walked.source,
                case.test,
                timeout=CASE_TIMEOUT_SECONDS,
                memory_mb=CASE_MEMORY_MB,
            )
            variant.update(
                {
                    "status": (
                        "preserved" if obfuscated.passed else "regression"
                    ),
                    "source_sha256": digest,
                    "source_bytes": len(walked.source.encode("utf-8")),
                    "obfuscated": obfuscated.to_dict(),
                    "trace": [record.to_dict() for record in walked.records],
                }
            )
        except BaseException as error:
            # Continue with the other independent variants so one case yields
            # all available diagnostics in a single batch run.
            variant.update(
                {
                    "status": "obfuscator_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        variants.append(variant)

    preserved = all(
        variant["status"] == "preserved" for variant in variants
    )
    return {
        **common,
        "status": "preserved" if preserved else "variant_failure",
        "distinct_output_count": len(output_digests),
        "variants": variants,
    }


def _failure_summary(result: dict[str, object]) -> str:
    return (
        f"{result['dataset']}:{result['task']} status={result['status']} "
        f"baseline={result.get('baseline')} "
        f"distinct_outputs={result.get('distinct_output_count')} "
        f"variants={result.get('variants')}"
    )


@pytest.mark.long
def test_all_reference_programs_survive_three_independent_walks() -> None:
    cases = tuple(reference_cases())
    counts = Counter(case.dataset for case in cases)
    assert counts == {"humaneval_py": 164, "mbpp_py": 378}

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        results = tuple(executor.map(_run_reference_case, cases, chunksize=1))

    case_statuses = Counter(result["status"] for result in results)
    variant_statuses = Counter(
        variant["status"]
        for result in results
        for variant in result.get("variants", ())
    )
    distinct_outputs = Counter(
        result["distinct_output_count"]
        for result in results
        if "distinct_output_count" in result
    )
    print(
        "reference random-walk summary: "
        f"cases={dict(case_statuses)} "
        f"variants={dict(variant_statuses)} "
        f"distinct_outputs_per_case={dict(distinct_outputs)}",
        flush=True,
    )

    failures = [result for result in results if result["status"] != "preserved"]
    assert not failures, "\n" + "\n".join(map(_failure_summary, failures))
