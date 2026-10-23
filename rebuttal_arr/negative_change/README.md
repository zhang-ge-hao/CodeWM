# Practical Negative-Class Experiment

This directory contains a standalone rebuttal experiment for changing the
negative distribution used by the paper's WLLM AUROC evaluation.  It does not
modify or write to the original `src/`, `data/original/`, `data/task/`, or
`data/result/` trees.  Every generated artifact is written below this
directory's `data/` folder.

## Scope

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Watermark: WLLM
- Datasets: HumanEval-Python (164 tasks), MBPP-Python (378 tasks)
- WLLM grid: 5 deltas x 3 gammas, temperature 1.0, 5-gram
- Metric: AUROC only

The three positive distributions already saved by the paper are:

1. `clean_wllm`
2. `pyminify_wllm`
3. `pyminifier_wllm`

Each is compared with all four new negative distributions:

1. `clean_no_wm_llm`
2. `benchmark_reference`
3. `pyminify_no_wm_llm`
4. `pyminifier_no_wm_llm`

This gives 12 new AUROCs per condition and 360 rows over 2 datasets x 15
conditions.  The paper's 90 standard-normal AUROCs are already present; they
can be normalized into the new schema as an optional, read-only step.

## Environment

Use the original environment:

```bash
~/conda/envs/watermarking/bin/python
```

Required components are checked by `validate` and are never installed by this
code:

- torch 2.5.x
- transformers 4.47.x
- scipy, scikit-learn, nltk
- a locally cached LLaMA-3.1-8B-Instruct tokenizer
- `pyminify` and `pyminifier` in the watermarking environment

Model weights and CUDA are not required.  Normal runs load the tokenizer with
`local_files_only=True`.

## Commands

Run commands from any working directory:

```bash
PY=~/conda/envs/watermarking/bin/python

$PY rebuttal_arr/negative_change/run.py validate
$PY rebuttal_arr/negative_change/run.py prepare --dataset all
$PY rebuttal_arr/negative_change/run.py score --all
$PY rebuttal_arr/negative_change/run.py aggregate
```

Scoring is split into 285 independently writable shards of at most 30 benchmark
tasks each.  A dataset/config selector runs all of that config's shards
sequentially:

```bash
$PY rebuttal_arr/negative_change/run.py score \
  --dataset humaneval_py --config 001

# Run just one local shard for debugging.
$PY rebuttal_arr/negative_change/run.py score \
  --dataset humaneval_py --config 001 --shard-index 0

# Production job-array entry: a stable 0-based global shard index.
$PY rebuttal_arr/negative_change/run.py score --job-index 0
```

WLLM detection is CPU-bound because the original detector constructs token
greenlists repeatedly. Prefer the 285 independent shards
(`--job-index 0..284`) over running `score --all` in one process. Configure
array concurrency and wall time for the target compute environment.

The global index is ordered by dataset, then config ID, then local part:

| Global indices | Dataset | Configs | Parts per config | Task slices |
| --- | --- | --- | ---: | --- |
| 0..89 | `humaneval_py` | 001..015 | 6 | five 30-task parts, then 14 |
| 90..284 | `mbpp_py` | 001..015 | 13 | twelve 30-task parts, then 18 |

Thus job 0 is HumanEval/001 part 0 `[0:30]`, job 5 is HumanEval/001
part 5 `[150:164]`, job 90 is MBPP/001 part 0 `[0:30]`, and job 102 is
MBPP/001 part 12 `[360:378]`.  A typical bounded-concurrency Slurm array is:

```bash
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --array=0-284%30

~/conda/envs/watermarking/bin/python \
  rebuttal_arr/negative_change/run.py score \
  --job-index "$SLURM_ARRAY_TASK_ID"
```

Each successful or already-existing shard prints one machine-readable final
line prefixed with `SHARD_RESULT`, including its job index, dataset, config,
part, task range, row count, status, and output path.

The included batch files reproduce the production submission and schedule
aggregation only after every array element succeeds:

```bash
ARRAY_JOB=$(sbatch --parsable rebuttal_arr/negative_change/score_array.sbatch)
sbatch --dependency="afterok:$ARRAY_JOB" \
  rebuttal_arr/negative_change/aggregate_after_array.sbatch

$PY rebuttal_arr/negative_change/summarize_shards.py --overwrite
```

Existing outputs are skipped by `prepare` and `score`.  Use `--overwrite`
explicitly to replace them.  Aggregate the existing N(0,1) baselines only when
they are needed for a joined table:

```bash
$PY rebuttal_arr/negative_change/run.py aggregate \
  --include-existing-normal --overwrite
```

## Input and pairing rules

- `data/task` is never read.  Its random seeds were regenerated after the
  saved experiments and do not match the keys that produced the results.
- Every negative is scored with the corresponding saved WLLM generation
  record's `custom_seed`, `gamma`, and `ngram_len`.
- HumanEval reference code is `prompt + canonical_solution`.
- MBPP reference code is its complete `canonical_solution`.
- The no-WM run is discovered from its saved temperature rather than assuming
  that directory `004` always means temperature 1.0.
- `passed` and `bad_trans` are not selection criteria, matching the paper's
  current detectability aggregation.
- The fixed positive cohort contains tasks with finite saved z-scores for all
  three positive variants.  Missing negative transformations reduce only that
  negative distribution and are reported explicitly.

## Outputs

```text
data/
├── manifest.json
├── negative_corpus/
│   ├── humaneval_py.jsonl
│   └── mbpp_py.jsonl
├── records/
│   ├── humaneval_py/{001..015}/part-{000..005}.jsonl
│   └── mbpp_py/{001..015}/part-{000..012}.jsonl
├── shard_summary.jsonl
├── slurm/
│   ├── score-ARRAY_JOB_TASK.{out,err}
│   └── aggregate-JOB.{out,err}
├── metrics_new.jsonl
└── metrics_normal_existing.jsonl  # optional
```

`negative_corpus/*.jsonl` stores the complete `g4d` and `solution` for all
four negative variants.  Negative code is stored once and reused for all 15
detector configurations.

`records/<dataset>/<config>/part-XXX.jsonl` stores, per task:

- the detection prompt and saved detector key/configuration;
- complete code and saved z-score for all three positive variants;
- newly computed z-scores for all four negatives;
- stable failure and detector-invalid metadata.

Every row also records the global job index, local part index, and half-open
task slice.  Before aggregation, all expected part files must exist and no
extra part file may exist.  Aggregation rejects duplicate, missing, unexpected,
or wrongly counted tasks, shard metadata mismatches, and detector parameters
that do not match the saved WLLM config.

`metrics_new.jsonl` is negative-major and contains one row per matrix cell:

```json
{
  "dataset": "humaneval_py",
  "config": "001",
  "delta": 0.5,
  "gamma": 0.1,
  "negative": "pyminify_no_wm_llm",
  "positive": "clean_wllm",
  "auroc": 0.57,
  "n_positive": 162,
  "n_negative": 161,
  "source": "new"
}
```

The manifest records input paths and SHA256 hashes, repository commit,
package/tool versions, construction rules, the complete discovered grid, and
the score shard size, ordering, per-dataset counts, and total job count.

`shard_summary.jsonl` has one validation row for each of the 285 subtasks. It
records the task slice, expected and actual row counts, output path, structural
validity, and finite/invalid/missing score counts for every positive and
negative class.

## Tests

```bash
~/conda/envs/watermarking/bin/python -m unittest discover \
  -s rebuttal_arr/negative_change/tests -v
```

The tests validate source coverage and ID mapping, all shard-index boundaries,
missing/extra/duplicate shard rejection, the full 4 x 3 aggregation, and exact
reproduction of saved clean and obfuscated WLLM z-scores.
