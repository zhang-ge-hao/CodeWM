"""Read and validate the inputs for the empirical-negative experiment.

This module deliberately treats the original experiment tree as read-only.  It
discovers runs from record contents (rather than assuming that a run number has
a particular meaning), normalises benchmark task identifiers, and fails early
when an input is incomplete or internally inconsistent.

No path under ``data/task`` is referenced or read here.  The detector key for a
task must always come from its persisted WLLM result record.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "data" / "result"
ORIGINAL_ROOT = REPO_ROOT / "data" / "original"

MODEL_SLUG = "Llama31Instruct8B"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
WATERMARK = "wllm"
SUPPORTED_DATASETS = ("humaneval_py", "mbpp_py")
EXPECTED_CONFIG_IDS = tuple(f"{number:03d}" for number in range(1, 16))
EXPECTED_DELTAS = (0.5, 1.0, 2.0, 3.0, 4.0)
EXPECTED_GAMMAS = (0.1, 0.25, 0.5)
EXPECTED_TEMPERATURE = 1.0
EXPECTED_NGRAM_LEN = 5


class SourceDataError(ValueError):
    """Raised when an experiment input is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    original_file: Path
    original_task_prefix: str
    expected_count: int
    human_eval_style: bool


DATASET_SPECS: Mapping[str, DatasetSpec] = {
    "humaneval_py": DatasetSpec(
        name="humaneval_py",
        original_file=ORIGINAL_ROOT / "humaneval-x_py.jsonl",
        original_task_prefix="Python/",
        expected_count=164,
        human_eval_style=True,
    ),
    "mbpp_py": DatasetSpec(
        name="mbpp_py",
        original_file=ORIGINAL_ROOT / "mbppp_py.jsonl",
        original_task_prefix="Mbpp/",
        expected_count=378,
        human_eval_style=False,
    ),
}


@dataclass(frozen=True)
class ReferenceRecord:
    """A benchmark reference in the text form passed to the detector."""

    task_name: str
    source_task_id: str
    prompt: str
    canonical_solution: str
    solution: str
    g4d: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "source_task_id": self.source_task_id,
            "prompt": self.prompt,
            "canonical_solution": self.canonical_solution,
            "solution": self.solution,
            "g4d": self.g4d,
        }


@dataclass(frozen=True)
class WllmConfig:
    dataset: str
    config_id: str
    directory: Path
    generate_path: Path
    obfuscate_path: Path
    delta: float
    gamma: float
    temperature: float
    ngram_len: int
    task_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "config_id": self.config_id,
            "directory": relative_to_repo(self.directory),
            "generate_path": relative_to_repo(self.generate_path),
            "obfuscate_path": relative_to_repo(self.obfuscate_path),
            "delta": self.delta,
            "gamma": self.gamma,
            "temperature": self.temperature,
            "ngram_len": self.ngram_len,
            "task_count": self.task_count,
        }


@dataclass(frozen=True)
class NoWmRun:
    dataset: str
    run_id: str
    directory: Path
    generate_path: Path
    temperature: float
    task_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "run_id": self.run_id,
            "directory": relative_to_repo(self.directory),
            "generate_path": relative_to_repo(self.generate_path),
            "temperature": self.temperature,
            "task_count": self.task_count,
        }


@dataclass(frozen=True)
class DatasetInputs:
    dataset: str
    references: Mapping[str, ReferenceRecord]
    wllm_configs: tuple[WllmConfig, ...]
    no_wm_run: NoWmRun
    task_names: tuple[str, ...]


def _require_dataset(dataset: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise SourceDataError(
            f"Unsupported dataset {dataset!r}; expected one of: {supported}"
        ) from exc


def relative_to_repo(path: Path) -> str:
    """Return a stable POSIX path for manifests."""

    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def iter_jsonl(
    path: Path | str,
    *,
    required_fields: Sequence[str] = (),
) -> Iterator[dict[str, Any]]:
    """Yield JSON objects with line-aware errors and optional field validation."""

    input_path = Path(path)
    if not input_path.is_file():
        raise SourceDataError(f"JSONL file does not exist: {input_path}")

    try:
        handle = input_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise SourceDataError(f"Cannot open JSONL file {input_path}: {exc}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SourceDataError(
                    f"Invalid JSON in {input_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise SourceDataError(
                    f"Expected a JSON object in {input_path}:{line_number}, "
                    f"got {type(value).__name__}"
                )
            missing = [field for field in required_fields if field not in value]
            if missing:
                raise SourceDataError(
                    f"Missing fields {missing!r} in {input_path}:{line_number}"
                )
            yield value


def read_jsonl(
    path: Path | str,
    *,
    required_fields: Sequence[str] = (),
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path, required_fields=required_fields))
    if not rows and not allow_empty:
        raise SourceDataError(f"JSONL file is empty: {path}")
    return rows


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def write_jsonl(
    path: Path | str,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write JSONL while protecting the read-only input trees."""

    output_path = Path(path)
    if _is_within(output_path, RESULT_ROOT) or _is_within(output_path, ORIGINAL_ROOT):
        raise SourceDataError(
            f"Refusing to write into an input tree: {output_path}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            for row_number, row in enumerate(rows, start=1):
                if not isinstance(row, Mapping):
                    raise TypeError(
                        f"JSONL row {row_number} is not a mapping: "
                        f"{type(row).__name__}"
                    )
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output_path)
    except BaseException:
        if temp_name is not None:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def index_by_task(
    rows: Iterable[Mapping[str, Any]],
    *,
    task_field: str = "task_name",
    context: str = "records",
) -> dict[str, dict[str, Any]]:
    """Index records by task, rejecting absent and duplicate task identifiers."""

    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        task_name = row.get(task_field)
        if not isinstance(task_name, str) or not task_name:
            raise SourceDataError(
                f"{context} row {row_number} has invalid {task_field!r}: {task_name!r}"
            )
        if task_name in indexed:
            raise SourceDataError(f"Duplicate task {task_name!r} in {context}")
        indexed[task_name] = dict(row)
    return indexed


def normalise_original_task_id(dataset: str, source_task_id: str) -> str:
    """Map benchmark IDs (``Python/N`` or ``Mbpp/N``) to result task IDs."""

    spec = _require_dataset(dataset)
    if not isinstance(source_task_id, str) or not source_task_id.startswith(
        spec.original_task_prefix
    ):
        raise SourceDataError(
            f"Unexpected task_id {source_task_id!r} in {dataset}; expected prefix "
            f"{spec.original_task_prefix!r}"
        )
    suffix = source_task_id[len(spec.original_task_prefix) :]
    if not suffix or not suffix.isdigit():
        raise SourceDataError(
            f"Unexpected numeric suffix in source task_id {source_task_id!r}"
        )
    return f"{dataset}/{suffix}"


def load_references(dataset: str) -> dict[str, ReferenceRecord]:
    """Load the complete benchmark-reference corpus for a supported dataset.

    HumanEval-X stores the function declaration in ``prompt`` and only its body
    in ``canonical_solution``; those fields are concatenated.  MBPP's canonical
    solution is already a complete function and is used on its own.
    """

    spec = _require_dataset(dataset)
    rows = read_jsonl(
        spec.original_file,
        required_fields=("task_id", "prompt", "canonical_solution"),
    )
    references: dict[str, ReferenceRecord] = {}
    for row in rows:
        source_task_id = row["task_id"]
        prompt = row["prompt"]
        canonical = row["canonical_solution"]
        if not all(isinstance(value, str) for value in (source_task_id, prompt, canonical)):
            raise SourceDataError(
                f"Reference {source_task_id!r} in {spec.original_file} has non-string "
                "task_id, prompt, or canonical_solution"
            )
        if not canonical:
            raise SourceDataError(
                f"Reference {source_task_id!r} has an empty canonical_solution"
            )
        task_name = normalise_original_task_id(dataset, source_task_id)
        if task_name in references:
            raise SourceDataError(
                f"Duplicate normalised reference task {task_name!r} in {spec.original_file}"
            )
        solution = prompt + canonical if spec.human_eval_style else canonical
        references[task_name] = ReferenceRecord(
            task_name=task_name,
            source_task_id=source_task_id,
            prompt=prompt,
            canonical_solution=canonical,
            solution=solution,
            g4d=solution,
        )

    if len(references) != spec.expected_count:
        raise SourceDataError(
            f"Expected {spec.expected_count} references for {dataset}, "
            f"found {len(references)}"
        )
    return references


_GENERATE_REQUIRED_FIELDS = (
    "id",
    "task_name",
    "dataset_name",
    "model_name",
    "watermarking",
    "language",
    "is_inst",
    "temperature",
    "delta",
    "gamma",
    "p4d",
    "g4d",
    "solution",
    "custom_seed",
    "ngram_len",
)


def load_generate_records(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load generation records, indexed by their persisted task name."""

    rows = read_jsonl(path, required_fields=_GENERATE_REQUIRED_FIELDS)
    return index_by_task(rows, context=str(path))


def _single_value(
    records: Mapping[str, Mapping[str, Any]],
    field: str,
    *,
    context: str,
) -> Any:
    values: list[Any] = []
    for record in records.values():
        value = record.get(field)
        if not any(value == seen and type(value) is type(seen) for seen in values):
            values.append(value)
    if len(values) != 1:
        raise SourceDataError(
            f"Expected one value for {field!r} in {context}, found {values!r}"
        )
    return values[0]


def _validate_common_generation_records(
    dataset: str,
    records: Mapping[str, Mapping[str, Any]],
    *,
    watermarking: str,
    context: str,
) -> None:
    spec = _require_dataset(dataset)
    if len(records) != spec.expected_count:
        raise SourceDataError(
            f"Expected {spec.expected_count} tasks in {context}, found {len(records)}"
        )

    expected_constants = {
        "dataset_name": dataset,
        "model_name": MODEL_NAME,
        "watermarking": watermarking,
        "language": "py",
        "is_inst": True,
    }
    for field, expected in expected_constants.items():
        actual = _single_value(records, field, context=context)
        if actual != expected or type(actual) is not type(expected):
            raise SourceDataError(
                f"Expected {field}={expected!r} in {context}, found {actual!r}"
            )

    for task_name, record in records.items():
        if not task_name.startswith(f"{dataset}/"):
            raise SourceDataError(
                f"Task {task_name!r} in {context} does not belong to {dataset}"
            )
        for text_field in ("p4d", "g4d", "solution"):
            if not isinstance(record[text_field], str):
                raise SourceDataError(
                    f"Task {task_name!r} has non-string {text_field} in {context}"
                )


def _numeric_run_directories(base: Path) -> list[Path]:
    if not base.is_dir():
        raise SourceDataError(f"Result directory does not exist: {base}")
    return sorted(
        path
        for path in base.iterdir()
        if path.is_dir() and len(path.name) == 3 and path.name.isdigit()
    )


def discover_wllm_configs(dataset: str) -> tuple[WllmConfig, ...]:
    """Discover and fully validate the 15 Llama-3.1 WLLM configurations."""

    spec = _require_dataset(dataset)
    base = RESULT_ROOT / f"{MODEL_SLUG}--{WATERMARK}--{dataset}"
    directories = _numeric_run_directories(base)
    ids = tuple(path.name for path in directories)
    if ids != EXPECTED_CONFIG_IDS:
        raise SourceDataError(
            f"Expected WLLM config directories {EXPECTED_CONFIG_IDS!r} in {base}, "
            f"found {ids!r}"
        )

    configs: list[WllmConfig] = []
    task_set: frozenset[str] | None = None
    seen_parameters: set[tuple[float, float]] = set()
    for directory in directories:
        generate_path = directory / "generate.jsonl"
        obfuscate_path = directory / "obfuscate.jsonl"
        if not obfuscate_path.is_file():
            raise SourceDataError(f"Missing obfuscation results: {obfuscate_path}")
        records = load_generate_records(generate_path)
        context = relative_to_repo(generate_path)
        _validate_common_generation_records(
            dataset, records, watermarking=WATERMARK, context=context
        )
        delta = _single_value(records, "delta", context=context)
        gamma = _single_value(records, "gamma", context=context)
        temperature = _single_value(records, "temperature", context=context)
        ngram_len = _single_value(records, "ngram_len", context=context)

        if not isinstance(delta, (int, float)) or isinstance(delta, bool):
            raise SourceDataError(f"Invalid delta {delta!r} in {context}")
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool):
            raise SourceDataError(f"Invalid gamma {gamma!r} in {context}")
        if not math.isclose(float(temperature), EXPECTED_TEMPERATURE):
            raise SourceDataError(
                f"Expected temperature={EXPECTED_TEMPERATURE} in {context}, "
                f"found {temperature!r}"
            )
        if ngram_len != EXPECTED_NGRAM_LEN:
            raise SourceDataError(
                f"Expected ngram_len={EXPECTED_NGRAM_LEN} in {context}, "
                f"found {ngram_len!r}"
            )
        for task_name, record in records.items():
            seed = record["custom_seed"]
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise SourceDataError(
                    f"Task {task_name!r} has invalid custom_seed {seed!r} in {context}"
                )

        current_task_set = frozenset(records)
        if task_set is None:
            task_set = current_task_set
        elif current_task_set != task_set:
            missing = sorted(task_set - current_task_set)
            extra = sorted(current_task_set - task_set)
            raise SourceDataError(
                f"Task-set mismatch in {context}: missing={missing[:5]!r}, "
                f"extra={extra[:5]!r}"
            )

        pair = (float(delta), float(gamma))
        if pair in seen_parameters:
            raise SourceDataError(f"Duplicate WLLM parameter pair {pair!r} in {base}")
        seen_parameters.add(pair)
        configs.append(
            WllmConfig(
                dataset=dataset,
                config_id=directory.name,
                directory=directory,
                generate_path=generate_path,
                obfuscate_path=obfuscate_path,
                delta=float(delta),
                gamma=float(gamma),
                temperature=float(temperature),
                ngram_len=int(ngram_len),
                task_count=len(records),
            )
        )

    expected_parameters = {
        (delta, gamma) for delta in EXPECTED_DELTAS for gamma in EXPECTED_GAMMAS
    }
    if seen_parameters != expected_parameters:
        raise SourceDataError(
            f"WLLM parameter grid mismatch for {dataset}: "
            f"missing={sorted(expected_parameters - seen_parameters)!r}, "
            f"extra={sorted(seen_parameters - expected_parameters)!r}"
        )
    if task_set is None or len(task_set) != spec.expected_count:
        raise SourceDataError(f"No complete WLLM task set found for {dataset}")
    return tuple(configs)


def discover_no_wm_run(
    dataset: str,
    *,
    temperature: float = EXPECTED_TEMPERATURE,
) -> NoWmRun:
    """Find the unique no-WM run whose records have the requested temperature."""

    _require_dataset(dataset)
    base = RESULT_ROOT / f"{MODEL_SLUG}--no_wm--{dataset}"
    candidates: list[tuple[Path, dict[str, dict[str, Any]], float]] = []
    for directory in _numeric_run_directories(base):
        generate_path = directory / "generate.jsonl"
        records = load_generate_records(generate_path)
        context = relative_to_repo(generate_path)
        _validate_common_generation_records(
            dataset, records, watermarking="no_wm", context=context
        )
        actual_temperature = _single_value(records, "temperature", context=context)
        if not isinstance(actual_temperature, (int, float)) or isinstance(
            actual_temperature, bool
        ):
            raise SourceDataError(
                f"Invalid temperature {actual_temperature!r} in {context}"
            )
        if math.isclose(float(actual_temperature), float(temperature)):
            candidates.append((directory, records, float(actual_temperature)))

    if len(candidates) != 1:
        candidate_ids = [directory.name for directory, _, _ in candidates]
        raise SourceDataError(
            f"Expected exactly one no-WM run at temperature={temperature} for "
            f"{dataset}, found {candidate_ids!r}"
        )
    directory, records, actual_temperature = candidates[0]
    return NoWmRun(
        dataset=dataset,
        run_id=directory.name,
        directory=directory,
        generate_path=directory / "generate.jsonl",
        temperature=actual_temperature,
        task_count=len(records),
    )


def load_obfuscation_records(
    config: WllmConfig,
    *,
    names: Sequence[str] = ("pyminify", "pyminifier"),
) -> dict[str, dict[str, dict[str, Any]]]:
    """Index persisted positive obfuscations as ``task -> obfuscator -> record``."""

    generate_records = load_generate_records(config.generate_path)
    id_to_task = {record["id"]: task for task, record in generate_records.items()}
    rows = read_jsonl(
        config.obfuscate_path,
        required_fields=(
            "id",
            "gen_task_id",
            "obf_name",
            "p4d",
            "g4d",
            "solution",
            "z_score",
        ),
        allow_empty=True,
    )
    allowed = set(names)
    indexed: dict[str, dict[str, dict[str, Any]]] = {
        task: {} for task in generate_records
    }
    for row_number, row in enumerate(rows, start=1):
        gen_task_id = row["gen_task_id"]
        try:
            task_name = id_to_task[gen_task_id]
        except KeyError as exc:
            raise SourceDataError(
                f"Unknown gen_task_id {gen_task_id!r} at "
                f"{config.obfuscate_path}:{row_number}"
            ) from exc
        name = row["obf_name"]
        if name not in allowed:
            continue
        if name in indexed[task_name]:
            raise SourceDataError(
                f"Duplicate {name!r} record for {task_name!r} in "
                f"{config.obfuscate_path}"
            )
        indexed[task_name][name] = row
    return indexed


def load_dataset_inputs(dataset: str) -> DatasetInputs:
    """Discover all source paths and verify that their task universes agree."""

    references = load_references(dataset)
    configs = discover_wllm_configs(dataset)
    no_wm_run = discover_no_wm_run(dataset)
    no_wm_records = load_generate_records(no_wm_run.generate_path)
    first_wllm_records = load_generate_records(configs[0].generate_path)

    reference_tasks = set(references)
    wllm_tasks = set(first_wllm_records)
    no_wm_tasks = set(no_wm_records)
    if not (reference_tasks == wllm_tasks == no_wm_tasks):
        raise SourceDataError(
            f"Task universes differ for {dataset}: references={len(reference_tasks)}, "
            f"WLLM={len(wllm_tasks)}, no-WM={len(no_wm_tasks)}"
        )

    for task_name in sorted(wllm_tasks):
        wllm_record = first_wllm_records[task_name]
        no_wm_record = no_wm_records[task_name]
        if wllm_record["p4d"] != no_wm_record["p4d"]:
            raise SourceDataError(
                f"Detection prompt mismatch between WLLM and no-WM for {task_name}"
            )

    return DatasetInputs(
        dataset=dataset,
        references=references,
        wllm_configs=configs,
        no_wm_run=no_wm_run,
        task_names=tuple(sorted(reference_tasks, key=_task_sort_key)),
    )


def _task_sort_key(task_name: str) -> tuple[str, int | str]:
    dataset, _, suffix = task_name.rpartition("/")
    return (dataset, int(suffix) if suffix.isdigit() else suffix)


def validate_inputs(dataset: str | None = None) -> dict[str, Any]:
    """Validate one or both datasets and return compact manifest-ready metadata."""

    datasets = (dataset,) if dataset is not None else SUPPORTED_DATASETS
    report: dict[str, Any] = {
        "repo_root": REPO_ROOT.as_posix(),
        "model": MODEL_NAME,
        "watermark": WATERMARK,
        "datasets": {},
    }
    for dataset_name in datasets:
        inputs = load_dataset_inputs(dataset_name)
        report["datasets"][dataset_name] = {
            "task_count": len(inputs.task_names),
            "reference_path": relative_to_repo(
                DATASET_SPECS[dataset_name].original_file
            ),
            "no_wm_run": inputs.no_wm_run.to_dict(),
            "wllm_configs": [config.to_dict() for config in inputs.wllm_configs],
        }
    return report


__all__ = [
    "DATASET_SPECS",
    "EXPECTED_CONFIG_IDS",
    "MODEL_NAME",
    "MODEL_SLUG",
    "ORIGINAL_ROOT",
    "REPO_ROOT",
    "RESULT_ROOT",
    "SUPPORTED_DATASETS",
    "DatasetInputs",
    "DatasetSpec",
    "NoWmRun",
    "ReferenceRecord",
    "SourceDataError",
    "WllmConfig",
    "discover_no_wm_run",
    "discover_wllm_configs",
    "index_by_task",
    "iter_jsonl",
    "load_dataset_inputs",
    "load_generate_records",
    "load_obfuscation_records",
    "load_references",
    "normalise_original_task_id",
    "read_jsonl",
    "relative_to_repo",
    "validate_inputs",
    "write_jsonl",
]
