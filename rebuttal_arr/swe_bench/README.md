# SWE-bench Verified BM25-13K pilot

This directory contains a small, resumable repository-level generation and
evaluation experiment.

## Workflow

1. Load `SWE-bench/SWE-bench_Verified` and deterministically sample 50 of its
   500 test IDs.
2. Join those IDs to `princeton-nlp/SWE-bench_bm25_13K` and use only its
   preconstructed `text` prompt. The target patch is never exposed to the
   generator.
3. Adapt the official text prompt to one Qwen user message, disable thinking,
   and generate one case at a time with Qwen3.6-35B-A3B. Ablations explicitly
   select either no watermark or the repository's WLLM logits processor. WLLM
   uses a fixed 5-gram context and the 15 combinations from delta
   `{0.5,1,2,3,4}` and gamma `{0.1,0.25,0.5}`. The output allowance is 10,240
   tokens per case.
4. Extract the patch with SWE-bench's official `extract_diff` function and
   write the standard `instance_id`, `model_name_or_path`, and `model_patch`
   prediction records.
5. Immediately after each case is generated, queue an asynchronous, isolated
   `swebench.harness.run_evaluation --modal true` process for that case. Up to
   eight Modal workers overlap with the remaining sequential inference. Each
   worker uses a unique run ID and directory, constructs the official TestSpec,
   and writes its result directly to disk. After inference, the main process
   waits for every Modal worker before writing the aggregate report and exiting.
6. For WLLM runs, detect every completion from the exact prompt and generated
   token IDs used during inference. Per-case results are in `detections.jsonl`;
   `detection_metrics.json` and the final summary contain aggregate z-score,
   detection-rate, green-token metrics, and ideal AUROC against the standard
   normal null. Both all-generated and patch-apply-pass AUROCs are recorded;
   the latter mirrors the paper's exclusion of uncompilable samples. No-WM
   runs do not invoke the WLLM processor or detector.

The official `swebench.inference.run_llama` model-loading loop is not reused:
it is specific to Llama tokenizers, SWE-Llama checkpoints, and fixed Llama
device maps. Its BM25 prompt data, output schema, diff extraction, and Modal
evaluation path are reused directly.

## Reproducibility and recovery

- The selection seed is fixed and `data/selection.json` proves that sampling
  starts from Verified rather than the original SWE-bench test set.
- Every Slurm job writes to `data/job-$SLURM_JOB_ID/`, so the earlier 1,024-token
  pilot cannot be mistaken for a resumed generation.
- Generation is appended per case to `generations.jsonl`; each
  `evaluations/case-NNN/` directory archives one prediction, hash, status,
  stdout/stderr, official single-case report, and official harness logs.
- `predictions.jsonl` contains the aggregate official harness input. Modal logs
  and the final aggregate official report are written in the same job directory.
- The parent retains the loaded model throughout. Once sequential inference is
  done, a single shared heartbeat keeps both GPUs active while the main process
  waits for any remaining Modal workers.

Submit from this directory so relative Slurm logs remain under `data/`:

```bash
sbatch run.sbatch
```

The ablation launcher writes the no-watermark result to
`data/ablation/non-wm/` and each WLLM result to a directory whose name encodes
delta, gamma, and the fixed 5-gram setting. Submit the three validation jobs
first:

```bash
./submit_ablation.sh smoke
```

Only after those results pass the solve-rate and detection checks, submit the
other 13 WLLM combinations:

```bash
./submit_ablation.sh remaining
```

The script discovers environments through `CONDA_ENV_ROOT`, or through
`$PI_WORKDIR/conda/envs` when that variable is unset. It contains no account,
user, email, or site-specific absolute paths. To resume a deliberately reused
run directory after a scheduler interruption, submit with the same
`EXPERIMENT_DATA_DIR` and `EXPERIMENT_RUN_ID`; otherwise each job is isolated.
