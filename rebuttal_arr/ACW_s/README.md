# ACW-s reproduction

This directory contains a traceable reproduction of the ACW-s oracle variant
from *Practical and Effective Code Watermarking for Large Language Models*.

## Layout

- `upstream/` is an unmodified snapshot of
  `TimeLovercc/code-watermark@4291a8e07abda0bc20560626f98340a90059be67`.
- `acw_s.py` is a small audited implementation of the same generation and
  detection primitives for later Python-Minifier experiments.
- `run_upstream.py` runs the official pipeline while disabling only optional
  WandB/ipdb integration and unauthenticated-Hugging-Face failures. It also
  repairs the upstream parser's unintended lower-casing of model IDs and paths.
  For the rebuttal analysis it retains prompt-only tasks as functional failures
  with zero watermark evidence instead of silently dropping them.
- `run_original_acws.sbatch` reproduces the original unmodified-code baseline
  on HumanEval and MBPP with one A100.
- `python_minifier_cpu.py` applies Python-Minifier to all 20 completions for
  all 164 HumanEval tasks and reruns their functional tests in four CPU shards.
- `compare_results.py` compares the output against paper Table 1.

The vendored source identity and checksums are in `UPSTREAM_PROVENANCE.json`.
The upstream repository had no `LICENSE` file at the pinned commit; keep this
copy limited to reproducibility work.

## Reproduction settings

Both tasks use OpenCoder-1.5B-Instruct, 20 completions per problem, BF16,
`temperature=0.2`, `top_p=0.95`, `top_k=0`, `delta=2`, `gamma=0.5`, seed 42,
and maximum total sequence length 2048. The official evaluator detects the
first non-empty completion for each problem and compares its z-score against
the human reference solution; Pass@1 and Pass@10 use all 20 completions.
The pinned upstream code silently drops a complete task if all 20 postprocessed
completions are prompt-only. The wrapper changes only that aggregation policy:
such tasks remain in the 164-task denominator with Pass@k=0 and machine z=0.

The entropy threshold is 1.2 for HumanEval and 0.3 for MBPP. These are the
dataset-specific thresholds reported in the paper's SWEET threshold-selection
appendix and are needed to test whether the ACW-s Table 1 result used the same
selection. The official shell runner instead defaults to 1.2 globally. This
distinction is recorded explicitly rather than treating 0.3 as a documented
ACW-s-specific optimum.

Paper Table 1 targets for OpenCoder-1.5B-Instruct are:

| Dataset | Pass@1 | Pass@10 | AUROC | TPR@5%FPR |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | 64.13 | 76.39 | 93.38 | 61.87 |
| MBPP | 40.64 | 48.33 | 88.91 | 54.80 |

## Unity

Submit from the repository root so the Slurm log paths resolve correctly:

```bash
mkdir -p rebuttal_arr/ACW_s/results
sbatch rebuttal_arr/ACW_s/run_original_acws.sbatch
```

The job defaults to `$HOME/conda/envs/watermarking/bin/python` and stores model
and dataset downloads under `$HF_HOME` or `$HOME/.cache/huggingface`. Every run
writes to `results/job-$SLURM_JOB_ID/`; `comparison.json` is created after both
datasets finish.

To skip a task in an already running sequential job before that task starts,
create `SKIP_HUMANEVAL` or `SKIP_MBPP` in that job's result root. The wrapper
and comparison report both recognize these markers.

Run the algorithm-equivalence checks without model generation using:

```bash
$HOME/conda/envs/watermarking/bin/python rebuttal_arr/ACW_s/smoke_test.py
```

## Python-Minifier CPU stage

The completed official run saved only 160 task files. The CPU stage restores
task indices 122, 137, 144, and 149 as 20 prompt-only completions per task,
which is the postprocessed output implied by the official evaluator, and marks
their baseline executions as failures. It therefore always validates exactly
164 tasks and 3,280 completions without filtering on generation or test status.

Submit the four shards and a dependent aggregation job with the source results
and a new output root exported through `ACWS_SOURCE_RESULTS` and
`ACWS_CPU_OUTPUT_ROOT`. The resulting `metrics.json` reports inclusive
Pass@1/Pass@10, transformation coverage, and functional regressions.
