#!/usr/bin/env python3
"""Generate BM25-13K patches for Verified tasks and evaluate them on Modal.

The retrieval prompt, diff extraction, prediction schema, and evaluation harness
come from SWE-bench.  Only the Qwen text-generation adapter is implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BM25_DATASET = "princeton-nlp/SWE-bench_bm25_13K"
VERIFIED_DATASET = "SWE-bench/SWE-bench_Verified"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
PREDICTION_MODEL_PREFIX = "Qwen3.6-35B-A3B--BM25-13K"
NGRAM_LEN = 5
SCHEMA_VERSION = 4

LOGGER = logging.getLogger("swe_bench_experiment")


@dataclass(frozen=True)
class GenerationSettings:
    model_id: str
    watermarking: str
    prompt_transport: str
    max_input_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    gamma: float | None
    delta: float | None
    ngram_len: int
    watermark_key: int
    z_threshold: float
    generation_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=0x1352766)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--watermarking",
        choices=("none", "wllm"),
        default="wllm",
        help="Generate without a watermark or with the repository WLLM scheme.",
    )
    parser.add_argument(
        "--prompt-transport",
        choices=("qwen_chat", "raw"),
        default="qwen_chat",
        help="How the official BM25 text is serialized for Qwen.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=10240)
    parser.add_argument("--modal-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--watermark-key", type=int, default=15485863)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--modal-timeout", type=int, default=1800)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--swebench-python", type=Path, required=True)
    parser.add_argument("--heartbeat-interval", type=float, default=15.0)
    parser.add_argument("--heartbeat-busy-seconds", type=float, default=2.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-modal", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.sample_size <= 500:
        parser.error("--sample-size must be in [1, 500]")
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        parser.error("token limits must be positive")
    if args.modal_workers <= 0:
        parser.error("--modal-workers must be positive")
    if args.watermarking == "wllm":
        if not 0.0 < args.gamma < 1.0:
            parser.error("--gamma must be in (0, 1) for WLLM")
        if args.delta <= 0:
            parser.error("--delta must be positive for WLLM")
    if args.z_threshold <= 0:
        parser.error("--z-threshold must be positive")
    if args.heartbeat_interval <= 0 or args.heartbeat_busy_seconds <= 0:
        parser.error("heartbeat timings must be positive")
    if not args.swebench_python.is_file():
        parser.error(f"SWE-bench Python does not exist: {args.swebench_python}")
    return args


def configure_logging(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(data_dir / "run.log")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def stable_seed(namespace: str, seed: int, instance_id: str) -> int:
    payload = f"{namespace}\0{seed}\0{instance_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_successful_generations(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from error
            if row.get("status") == "ok":
                rows[row["instance_id"]] = row
    return rows


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def format_hparam(value: float) -> str:
    return f"{value:g}"


def prediction_model_name(settings: GenerationSettings) -> str:
    if settings.watermarking == "none":
        return f"{PREDICTION_MODEL_PREFIX}--non-wm"
    assert settings.gamma is not None and settings.delta is not None
    return (
        f"{PREDICTION_MODEL_PREFIX}--WLLM"
        f"--delta-{format_hparam(settings.delta)}"
        f"--gamma-{format_hparam(settings.gamma)}"
        f"--ngram-{settings.ngram_len}"
    )


def prepare_cases(
    *, data_dir: Path, sample_size: int, selection_seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    LOGGER.info("Loading %s test split from the Hugging Face cache", VERIFIED_DATASET)
    verified = load_dataset(VERIFIED_DATASET, split="test")
    LOGGER.info("Loading %s test split from the Hugging Face cache", BM25_DATASET)
    bm25 = load_dataset(BM25_DATASET, split="test")

    if len(verified) != 500:
        raise RuntimeError(f"Expected 500 Verified cases, found {len(verified)}")

    verified_by_id = {row["instance_id"]: row for row in verified}
    bm25_by_id = {row["instance_id"]: row for row in bm25}
    intersection = sorted(set(verified_by_id) & set(bm25_by_id))
    if len(intersection) != len(verified_by_id):
        missing = sorted(set(verified_by_id) - set(bm25_by_id))
        raise RuntimeError(f"Verified cases missing BM25 text: {missing[:5]}")

    manifest_path = data_dir / "selection.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = (sample_size, selection_seed, VERIFIED_DATASET, BM25_DATASET)
        actual = (
            manifest.get("sample_size"),
            manifest.get("selection_seed"),
            manifest.get("selection_dataset"),
            manifest.get("prompt_dataset"),
        )
        if actual != expected:
            raise RuntimeError(
                "Existing selection.json does not match requested configuration: "
                f"expected={expected!r}, actual={actual!r}"
            )
        selected_ids = manifest["instance_ids"]
    else:
        selected_ids = random.Random(selection_seed).sample(
            sorted(verified_by_id), sample_size
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "selection_dataset": VERIFIED_DATASET,
            "selection_split": "test",
            "prompt_dataset": BM25_DATASET,
            "prompt_split": "test",
            "selection_population": len(verified_by_id),
            "verified_bm25_intersection": len(intersection),
            "sample_size": sample_size,
            "selection_seed": selection_seed,
            "instance_ids": selected_ids,
            "cases": [
                {
                    "instance_id": instance_id,
                    "repo": verified_by_id[instance_id]["repo"],
                    "difficulty": verified_by_id[instance_id].get("difficulty"),
                    "bm25_prompt_characters": len(bm25_by_id[instance_id]["text"]),
                }
                for instance_id in selected_ids
            ],
        }
        atomic_write_json(manifest_path, manifest)

    if len(selected_ids) != sample_size or len(set(selected_ids)) != sample_size:
        raise RuntimeError("Selection manifest does not contain the requested unique IDs")

    cases: list[dict[str, Any]] = []
    identity_fields = ("repo", "base_commit", "problem_statement", "version")
    for instance_id in selected_ids:
        if instance_id not in verified_by_id:
            raise RuntimeError(f"Selection ID is not in Verified: {instance_id}")
        verified_row = verified_by_id[instance_id]
        bm25_row = bm25_by_id[instance_id]
        for field in identity_fields:
            if verified_row[field] != bm25_row[field]:
                raise RuntimeError(f"Dataset mismatch for {instance_id}: {field}")
        # Deliberately expose only the preconstructed retrieval text to generation.
        cases.append(
            {
                "instance_id": instance_id,
                "repo": verified_row["repo"],
                "difficulty": verified_row.get("difficulty"),
                "text": bm25_row["text"],
            }
        )
    return cases, manifest


def cuda_smoke_test() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot initialize CUDA")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"This run requires exactly 2 visible GPUs; found {torch.cuda.device_count()}"
        )
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        if properties.total_memory < 75 * 1024**3:
            raise RuntimeError(
                f"cuda:{index} has only {properties.total_memory / 1024**3:.1f} GiB"
            )
        left = torch.randn((1024, 1024), device=index, dtype=torch.bfloat16)
        right = torch.randn((1024, 1024), device=index, dtype=torch.bfloat16)
        result = left @ right
        torch.cuda.synchronize(index)
        if not torch.isfinite(result).all().item():
            raise RuntimeError(f"Non-finite BF16 matmul result on cuda:{index}")
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return devices


def normalize_device(value: Any) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    text = str(value)
    if text.isdigit():
        return f"cuda:{text}"
    return text


def load_qwen(model_id: str, allow_download: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOGGER.info("Loading text-only model %s across two GPUs", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=not allow_download,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "72GiB", 1: "72GiB", "cpu": "96GiB"},
        low_cpu_mem_usage=True,
        local_files_only=not allow_download,
    ).eval()

    device_map = {
        key: normalize_device(value) for key, value in model.hf_device_map.items()
    }
    placements = set(device_map.values())
    if "cpu" in placements or "disk" in placements:
        raise RuntimeError(f"Model was offloaded outside GPUs: {placements}")
    if not {"cuda:0", "cuda:1"}.issubset(placements):
        raise RuntimeError(f"Model is not distributed across both GPUs: {placements}")
    LOGGER.info("Model loaded with placements=%s", sorted(placements))
    return model, tokenizer, device_map


def render_prompt(tokenizer, text: str, transport: str) -> tuple[str, bool]:
    """Render one prompt and return whether tokenizer special tokens are needed."""

    if transport == "raw":
        return text, True
    messages = [{"role": "user", "content": text}]
    return (
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        ),
        False,
    )


def prepare_prompt_cases(
    tokenizer, cases: list[dict[str, Any]], settings: GenerationSettings
) -> list[dict[str, Any]]:
    """Render and count prompts without truncating BM25 context."""

    prepared = []
    for selection_index, case in enumerate(cases):
        rendered, add_special_tokens = render_prompt(
            tokenizer, case["text"], settings.prompt_transport
        )
        input_tokens = len(
            tokenizer(rendered, add_special_tokens=add_special_tokens)["input_ids"]
        )
        if input_tokens > settings.max_input_tokens:
            raise RuntimeError(
                f"{case['instance_id']} has {input_tokens} Qwen tokens, above the "
                f"configured limit {settings.max_input_tokens}; refusing to truncate "
                "BM25 context"
            )
        prepared.append(
            {
                **case,
                "_rendered_prompt": rendered,
                "_add_special_tokens": add_special_tokens,
                "_input_tokens": input_tokens,
                "_selection_index": selection_index,
            }
        )

    return prepared


def create_watermark_components(tokenizer, settings: GenerationSettings):
    from transformers import LogitsProcessorList
    from src._sweet import WatermarkDetector, WatermarkLogitsProcessor

    # Match the repository implementation exactly: its vocabulary values are
    # used to determine partition size, even if the model has padded logits.
    vocab = list(tokenizer.get_vocab().values())
    if settings.watermarking == "none":
        return None, None, len(vocab)

    assert settings.gamma is not None and settings.delta is not None
    processor = WatermarkLogitsProcessor(
        vocab=vocab,
        gamma=settings.gamma,
        delta=settings.delta,
        seeding_scheme="n_grams",
        ngram_len=settings.ngram_len,
        hash_key=settings.watermark_key,
        select_green_tokens=True,
    )
    detector = WatermarkDetector(
        vocab=vocab,
        gamma=settings.gamma,
        delta=settings.delta,
        seeding_scheme="n_grams",
        ngram_len=settings.ngram_len,
        hash_key=settings.watermark_key,
        select_green_tokens=True,
        tokenizer=tokenizer,
        z_threshold=settings.z_threshold,
    )
    return LogitsProcessorList([processor]), detector, len(vocab)


def normalize_detection_result(
    raw: dict[str, Any], *, settings: GenerationSettings, elapsed: float
) -> dict[str, Any]:
    invalid = bool(raw.get("invalid", False))
    result = {
        "enabled": True,
        "scope": "full_completion_exact_generation_token_ids",
        "z_threshold": settings.z_threshold,
        "invalid": invalid,
        "elapsed_seconds": round(elapsed, 3),
        "prediction": bool(raw.get("prediction", False)) if not invalid else False,
    }
    for key in ("num_tokens_scored", "num_green_tokens"):
        value = raw.get(key)
        result[key] = None if value is None else int(value)
    for key in ("green_fraction", "z_score", "p_value", "confidence"):
        value = raw.get(key)
        result[key] = None if value is None else float(value)
    return result


def detect_completion(
    *, detector, prompt_ids, generated_ids: list[int], settings: GenerationSettings
) -> dict[str, Any] | None:
    if detector is None:
        return None

    import torch

    prefix = prompt_ids.detach().cpu().to(dtype=torch.long)
    suffix = torch.tensor(generated_ids, dtype=torch.long)
    tokenized_text = torch.cat((prefix, suffix), dim=0)
    started = time.monotonic()
    raw = detector.detect(
        tokenized_text=tokenized_text,
        tokenized_prefix=prefix,
        return_green_token_mask=False,
    )
    elapsed = time.monotonic() - started
    return normalize_detection_result(raw, settings=settings, elapsed=elapsed)


def first_subsequence_end(values: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    for start in range(len(values) - len(needle) + 1):
        if values[start : start + len(needle)] == needle:
            return start + len(needle)
    return None


def trim_generated_ids(
    values: list[int],
    patch_end_ids: list[int],
    eos_ids: set[int],
) -> tuple[list[int], str]:
    """Exclude EOS and identify why one generated sequence stopped."""

    candidates: list[tuple[int, int, str]] = []
    patch_end = first_subsequence_end(values, patch_end_ids)
    if patch_end is not None:
        candidates.append((patch_end, 0, "patch_end"))
    for index, token_id in enumerate(values):
        if token_id in eos_ids:
            # Exclude Qwen special EOS tokens such as <|im_end|> from a diff.
            candidates.append((index, 1, "eos_token"))
            break
    if not candidates:
        return values, "max_new_tokens"
    end, _, reason = min(candidates)
    return values[:end], reason


def generate_one(
    *,
    model,
    tokenizer,
    case: dict[str, Any],
    settings: GenerationSettings,
    logits,
    detector,
    generation_index: int,
) -> dict[str, Any]:
    import torch
    from transformers import set_seed
    from swebench.inference.make_datasets.utils import extract_diff

    instance_id = case["instance_id"]
    case_seed = stable_seed("generation", settings.generation_seed, instance_id)
    set_seed(case_seed, deterministic=False)
    encoded = tokenizer(
        case["_rendered_prompt"],
        add_special_tokens=case["_add_special_tokens"],
        return_tensors="pt",
    )
    input_tokens = int(encoded["input_ids"].shape[1])
    if input_tokens != case["_input_tokens"]:
        raise RuntimeError(
            f"Token count changed for {instance_id}: {input_tokens} != "
            f"{case['_input_tokens']}"
        )
    prompt_ids = encoded["input_ids"][0].detach().cpu()

    input_device = model.get_input_embeddings().weight.device
    encoded = encoded.to(input_device)
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)

    started = time.monotonic()
    with torch.inference_mode():
        sequences = model.generate(
            **encoded,
            do_sample=True,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            max_new_tokens=settings.max_new_tokens,
            use_cache=True,
            logits_processor=logits,
            stop_strings=["</patch>"],
            tokenizer=tokenizer,
        )
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)
    elapsed = time.monotonic() - started

    generated_ids = sequences[0, input_tokens:].detach().cpu().tolist()
    peak_memory_gib = [
        round(torch.cuda.max_memory_allocated(index) / 1024**3, 3)
        for index in range(torch.cuda.device_count())
    ]
    del sequences, encoded

    eos = model.generation_config.eos_token_id
    if eos is None:
        eos_ids: set[int] = set()
    elif isinstance(eos, int):
        eos_ids = {eos}
    else:
        eos_ids = set(eos)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    patch_end_ids = tokenizer.encode("</patch>", add_special_tokens=False)
    generated_ids, stop_reason = trim_generated_ids(
        generated_ids, patch_end_ids, eos_ids
    )
    completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
    if "</patch>" in completion:
        stop_reason = "patch_end"
    model_patch = extract_diff(completion) or ""
    detection = detect_completion(
        detector=detector,
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        settings=settings,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "instance_id": instance_id,
        "model_name_or_path": prediction_model_name(settings),
        "model_patch": model_patch,
        "full_output": completion,
        "generated_token_ids": generated_ids,
        "input_tokens": input_tokens,
        "generated_tokens": len(generated_ids),
        "stop_reason": stop_reason,
        "hit_max_new_tokens": stop_reason == "max_new_tokens",
        "generation_index": generation_index,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_second": round(len(generated_ids) / elapsed, 3),
        "generation_seed": case_seed,
        "peak_gpu_memory_gib": peak_memory_gib,
        "watermarking": settings.watermarking,
        "watermark_key": (
            settings.watermark_key if settings.watermarking == "wllm" else None
        ),
        "gamma": settings.gamma,
        "delta": settings.delta,
        "ngram_len": settings.ngram_len,
        "detection": detection,
        "prompt_transport": settings.prompt_transport,
    }


def write_predictions(
    path: Path, selected_ids: list[str], rows: dict[str, dict[str, Any]]
) -> None:
    predictions = []
    for instance_id in selected_ids:
        if instance_id not in rows:
            continue
        row = rows[instance_id]
        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": row["model_name_or_path"],
                "model_patch": row["model_patch"],
            }
        )
    atomic_write_jsonl(path, predictions)


def write_detections(
    path: Path, selected_ids: list[str], rows: dict[str, dict[str, Any]]
) -> None:
    detections = []
    for instance_id in selected_ids:
        row = rows.get(instance_id)
        if row is None or row.get("detection") is None:
            continue
        detections.append(
            {
                "schema_version": SCHEMA_VERSION,
                "instance_id": instance_id,
                "watermarking": row.get("watermarking"),
                "gamma": row.get("gamma"),
                "delta": row.get("delta"),
                "ngram_len": row.get("ngram_len"),
                **row["detection"],
            }
        )
    atomic_write_jsonl(path, detections)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GPUHeartbeat:
    """Create brief activity on every allocated GPU while Modal is running."""

    def __init__(self, data_dir: Path, interval: float, busy_seconds: float) -> None:
        self.path = data_dir / "gpu_heartbeat.jsonl"
        self.interval = interval
        self.busy_seconds = busy_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="gpu-heartbeat", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(60.0, self.interval + 10.0))

    def _run(self) -> None:
        import torch

        try:
            tensors = []
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    left = torch.randn((4096, 4096), dtype=torch.bfloat16, device=index)
                    right = torch.randn((4096, 4096), dtype=torch.bfloat16, device=index)
                    output = torch.empty_like(left)
                    tensors.append((index, left, right, output))

            while not self.stop_event.is_set():
                cycle_start = time.monotonic()
                for index, left, right, output in tensors:
                    device_start = time.monotonic()
                    operations = 0
                    with torch.cuda.device(index), torch.inference_mode():
                        while time.monotonic() - device_start < self.busy_seconds:
                            for _ in range(8):
                                torch.mm(left, right, out=output)
                                operations += 1
                            torch.cuda.synchronize(index)
                    append_jsonl(
                        self.path,
                        {
                            "time_unix": time.time(),
                            "device": index,
                            "operations": operations,
                            "busy_seconds": round(time.monotonic() - device_start, 3),
                        },
                    )
                delay = max(0.0, self.interval - (time.monotonic() - cycle_start))
                self.stop_event.wait(delay)
        except Exception as error:  # Keep the Modal process observable on failure.
            self.error = f"{type(error).__name__}: {error}"
            append_jsonl(
                self.path,
                {"time_unix": time.time(), "status": "error", "error": self.error},
            )


def synthetic_single_report(
    instance_id: str, *, empty_patch: bool, error: bool
) -> dict[str, Any]:
    empty_ids = [instance_id] if empty_patch else []
    error_ids = [instance_id] if error and not empty_patch else []
    return {
        "total_instances": 1,
        "submitted_instances": 1,
        "completed_instances": 0,
        "resolved_instances": 0,
        "unresolved_instances": 0,
        "empty_patch_instances": len(empty_ids),
        "error_instances": len(error_ids),
        "completed_ids": [],
        "incomplete_ids": [],
        "empty_patch_ids": empty_ids,
        "submitted_ids": [instance_id],
        "resolved_ids": [],
        "unresolved_ids": [],
        "error_ids": error_ids,
        "schema_version": 2,
        "unstopped_instances": 0,
        "unstopped_containers": [],
        "unremoved_images": [],
    }


def outcome_from_report(instance_id: str, report: dict[str, Any]) -> str:
    for field, outcome in (
        ("resolved_ids", "resolved"),
        ("unresolved_ids", "unresolved"),
        ("empty_patch_ids", "empty_patch"),
        ("error_ids", "error"),
        ("incomplete_ids", "incomplete"),
    ):
        if instance_id in report.get(field, []):
            return outcome
    return "unknown"


def evaluation_case_dir(data_dir: Path, generation_index: int) -> Path:
    return data_dir / "evaluations" / f"case-{generation_index:03d}"


def prepare_async_evaluation(
    *,
    data_dir: Path,
    base_run_id: str,
    generation_index: int,
    instance_id: str,
    row: dict[str, Any],
) -> tuple[Path, str]:
    case_dir = evaluation_case_dir(data_dir, generation_index)
    case_dir.mkdir(parents=True, exist_ok=True)
    status_path = case_dir / "status.json"
    existing = json.loads(status_path.read_text()) if status_path.exists() else {}
    attempt = int(existing.get("attempt", 0)) + 1
    evaluation_run_id = (
        f"{base_run_id}-eval-{generation_index:03d}-attempt-{attempt:02d}"
    )
    predictions_path = case_dir / "predictions.jsonl"
    write_predictions(predictions_path, [instance_id], {instance_id: row})
    status = {
        "schema_version": SCHEMA_VERSION,
        "base_run_id": base_run_id,
        "evaluation_run_id": evaluation_run_id,
        "generation_index": generation_index,
        "instance_id": instance_id,
        "prediction_model_name": row["model_name_or_path"],
        "attempt": attempt,
        "prediction_sha256": sha256_file(predictions_path),
        "model_patch_nonempty": bool(row["model_patch"].strip()),
        "evaluation_state": "queued",
        "evaluation_complete": False,
        "queued_time_unix": time.time(),
    }
    atomic_write_json(status_path, status)
    return case_dir, evaluation_run_id


def evaluate_one_modal(
    *,
    swebench_python: Path,
    case_dir: Path,
    instance_id: str,
    evaluation_run_id: str,
    timeout: int,
) -> dict[str, Any]:
    status_path = case_dir / "status.json"
    status = json.loads(status_path.read_text())
    status["evaluation_state"] = "running"
    status["started_time_unix"] = time.time()
    atomic_write_json(status_path, status)

    predictions_path = case_dir / "predictions.jsonl"
    command = [
        str(swebench_python),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        VERIFIED_DATASET,
        "--split",
        "test",
        "--predictions_path",
        str(predictions_path.resolve()),
        "--instance_ids",
        instance_id,
        "--run_id",
        evaluation_run_id,
        "--timeout",
        str(timeout),
        "--modal",
        "true",
    ]
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["MODAL_BUILD_VALIDATION"] = "ignore"
    environment["CUDA_VISIBLE_DEVICES"] = ""

    LOGGER.info(
        "[eval %03d] Starting asynchronous Modal evaluation for %s",
        status["generation_index"],
        instance_id,
    )
    started = time.monotonic()
    modal_exit_code: int | None = None
    worker_error: str | None = None
    stdout_path = case_dir / f"modal-attempt-{status['attempt']:02d}.out.log"
    stderr_path = case_dir / f"modal-attempt-{status['attempt']:02d}.err.log"
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(
                command,
                cwd=case_dir,
                env=environment,
                stdout=stdout,
                stderr=stderr,
            )
            modal_exit_code = process.wait()
    except Exception as error:
        worker_error = f"{type(error).__name__}: {error}"
        LOGGER.exception(
            "[eval %03d] Modal worker failed for %s",
            status["generation_index"],
            instance_id,
        )
    elapsed = time.monotonic() - started

    prediction_name = status["prediction_model_name"]
    report_name = f"{prediction_name.replace('/', '__')}.{evaluation_run_id}.json"
    generated_report_path = case_dir / report_name
    if generated_report_path.exists():
        try:
            report = json.loads(generated_report_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            worker_error = f"{type(error).__name__}: {error}"
            report = synthetic_single_report(
                instance_id, empty_patch=False, error=True
            )
    else:
        report = synthetic_single_report(
            instance_id,
            empty_patch=not status["model_patch_nonempty"],
            error=bool(status["model_patch_nonempty"]),
        )
    atomic_write_json(case_dir / "official_report.json", report)
    outcome = outcome_from_report(instance_id, report)
    run_instance_log = (
        case_dir
        / "logs"
        / "run_evaluation"
        / evaluation_run_id
        / prediction_name.replace("/", "__")
        / instance_id
        / "run_instance.log"
    )
    status.update(
        {
            "evaluation_state": "complete",
            "evaluation_complete": True,
            "completed_time_unix": time.time(),
            "modal_exit_code": modal_exit_code,
            "modal_elapsed_seconds": round(elapsed, 3),
            "worker_error": worker_error,
            "outcome": outcome,
            "official_report_file": "official_report.json",
            "run_instance_log": (
                str(run_instance_log.relative_to(case_dir))
                if run_instance_log.exists()
                else None
            ),
        }
    )
    atomic_write_json(status_path, status)
    LOGGER.info(
        "[eval %03d] Modal result %s: %s exit_code=%s elapsed=%.1fs",
        status["generation_index"],
        instance_id,
        outcome,
        modal_exit_code,
        elapsed,
    )
    return status


def load_evaluation_statuses(data_dir: Path) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for path in sorted((data_dir / "evaluations").glob("case-*/status.json")):
        status = json.loads(path.read_text())
        status["_case_dir"] = str(path.parent)
        statuses[status["instance_id"]] = status
    return statuses


def aggregate_official_reports(
    *,
    data_dir: Path,
    run_id: str,
    selected_ids: list[str],
    prediction_name: str,
) -> dict[str, Any]:
    statuses = load_evaluation_statuses(data_dir)
    id_fields = (
        "completed_ids",
        "empty_patch_ids",
        "submitted_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
    )
    combined = {field: set() for field in id_fields}
    for instance_id in selected_ids:
        status = statuses.get(instance_id)
        if not status or not status.get("evaluation_complete"):
            continue
        report_path = Path(status["_case_dir"]) / "official_report.json"
        if not report_path.exists():
            combined["error_ids"].add(instance_id)
            combined["submitted_ids"].add(instance_id)
            continue
        report = json.loads(report_path.read_text())
        for field in id_fields:
            combined[field].update(report.get(field, []))

    submitted_ids = combined["submitted_ids"]
    incomplete_ids = set(selected_ids) - submitted_ids
    report = {
        "total_instances": len(selected_ids),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(combined["completed_ids"]),
        "resolved_instances": len(combined["resolved_ids"]),
        "unresolved_instances": len(combined["unresolved_ids"]),
        "empty_patch_instances": len(combined["empty_patch_ids"]),
        "error_instances": len(combined["error_ids"]),
        "completed_ids": sorted(combined["completed_ids"]),
        "incomplete_ids": sorted(incomplete_ids),
        "empty_patch_ids": sorted(combined["empty_patch_ids"]),
        "submitted_ids": sorted(submitted_ids),
        "resolved_ids": sorted(combined["resolved_ids"]),
        "unresolved_ids": sorted(combined["unresolved_ids"]),
        "error_ids": sorted(combined["error_ids"]),
        "schema_version": 2,
        "unstopped_instances": 0,
        "unstopped_containers": [],
        "unremoved_images": [],
        "aggregation": "union_of_official_single_case_reports",
    }
    atomic_write_json(data_dir / "aggregate_official_report.json", report)
    atomic_write_json(
        data_dir / f"{prediction_name}.{run_id}.json", report
    )
    return report


def log_final_report(report: dict[str, Any]) -> None:
    LOGGER.info(
        "Final Modal summary: submitted=%d completed=%d resolved=%d "
        "unresolved=%d empty=%d errors=%d incomplete=%d",
        report["submitted_instances"],
        report["completed_instances"],
        report["resolved_instances"],
        report["unresolved_instances"],
        report["empty_patch_instances"],
        report["error_instances"],
        len(report["incomplete_ids"]),
    )


def ideal_auroc(z_scores: list[float]) -> float | None:
    """AUROC against the analytical N(0, 1) null used by the paper.

    For one positive score Z and an independent negative N ~ N(0, 1), the
    pairwise ranking probability is P(Z > N) = Phi(Z). Averaging Phi(Z) over
    the empirical positive scores is the exact AUROC, without sampling a
    finite negative set or approximating the ROC curve on a threshold grid.
    """

    if not z_scores:
        return None
    import scipy.stats

    return statistics.fmean(float(scipy.stats.norm.cdf(score)) for score in z_scores)


def summarize_detections(
    rows: dict[str, dict[str, Any]],
    settings: GenerationSettings,
    patch_apply_pass_ids: set[str],
) -> dict[str, Any]:
    detections = {
        instance_id: row["detection"]
        for instance_id, row in rows.items()
        if isinstance(row.get("detection"), dict)
    }
    if settings.watermarking == "none":
        return {
            "enabled": False,
            "scope": None,
            "z_threshold": settings.z_threshold,
            "scored_cases": 0,
        }

    valid = {
        instance_id: result
        for instance_id, result in detections.items()
        if not result.get("invalid") and result.get("z_score") is not None
    }
    z_scores = [float(result["z_score"]) for result in valid.values()]
    patch_apply_z_scores = [
        float(valid[instance_id]["z_score"])
        for instance_id in sorted(patch_apply_pass_ids & set(valid))
    ]
    p_values = [float(result["p_value"]) for result in valid.values()]
    tokens_scored = sum(
        int(result["num_tokens_scored"]) for result in valid.values()
    )
    green_tokens = sum(
        int(result["num_green_tokens"]) for result in valid.values()
    )
    return {
        "enabled": True,
        "scope": "full_completion_exact_generation_token_ids",
        "gamma": settings.gamma,
        "delta": settings.delta,
        "ngram_len": settings.ngram_len,
        "z_threshold": settings.z_threshold,
        "scored_cases": len(detections),
        "valid_cases": len(valid),
        "invalid_cases": len(detections) - len(valid),
        "detected_cases": sum(
            bool(result.get("prediction")) for result in valid.values()
        ),
        "detection_rate": (
            sum(bool(result.get("prediction")) for result in valid.values())
            / len(valid)
            if valid
            else None
        ),
        "auroc_negative_distribution": "standard_normal_N(0,1)",
        "auroc_all_generated": ideal_auroc(z_scores),
        "auroc_patch_apply_pass": ideal_auroc(patch_apply_z_scores),
        "auroc_patch_apply_pass_cases": len(patch_apply_z_scores),
        "mean_z_score": statistics.fmean(z_scores) if z_scores else None,
        "median_z_score": statistics.median(z_scores) if z_scores else None,
        "min_z_score": min(z_scores) if z_scores else None,
        "max_z_score": max(z_scores) if z_scores else None,
        "mean_p_value": statistics.fmean(p_values) if p_values else None,
        "total_tokens_scored": tokens_scored,
        "total_green_tokens": green_tokens,
        "weighted_green_fraction": (
            green_tokens / tokens_scored if tokens_scored else None
        ),
    }


def build_summary(
    *,
    data_dir: Path,
    run_id: str,
    selected_ids: list[str],
    generations_path: Path,
    official_report: dict[str, Any] | None,
    settings: GenerationSettings,
) -> dict[str, Any]:
    from swebench.harness.constants import APPLY_PATCH_FAIL, APPLY_PATCH_PASS

    all_rows = read_successful_generations(generations_path)
    rows = {
        instance_id: all_rows[instance_id]
        for instance_id in selected_ids
        if instance_id in all_rows
    }
    statuses = load_evaluation_statuses(data_dir)
    apply_pass = 0
    apply_fail = 0
    outcomes: dict[str, int] = {}
    for instance_id in selected_ids:
        status = statuses.get(instance_id)
        if not status:
            continue
        outcome = status.get("outcome")
        if outcome:
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        relative_log = status.get("run_instance_log")
        if not relative_log:
            continue
        log_path = Path(status["_case_dir"]) / relative_log
        if not log_path.exists():
            continue
        text = log_path.read_text(errors="replace")
        if APPLY_PATCH_PASS in text:
            apply_pass += 1
        if APPLY_PATCH_FAIL in text:
            apply_fail += 1

    patch_apply_pass_ids = {
        instance_id
        for instance_id, status in statuses.items()
        if status.get("outcome") in {"resolved", "unresolved"}
    }
    detection_metrics = summarize_detections(
        rows, settings, patch_apply_pass_ids
    )
    resolved_instances = (
        int(official_report["resolved_instances"]) if official_report else None
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "selected_cases": len(selected_ids),
        "generated_cases": len(rows),
        "nonempty_patches": sum(
            bool(row.get("model_patch", "").strip()) for row in rows.values()
        ),
        "unified_diff_outputs": sum(
            (
                "diff --git " in row.get("model_patch", "")
                or "--- a/" in row.get("model_patch", "")
            )
            for row in rows.values()
        ),
        "patch_apply_pass_logs": apply_pass,
        "patch_apply_fail_logs": apply_fail,
        "hit_max_new_tokens": sum(
            bool(row.get("hit_max_new_tokens")) for row in rows.values()
        ),
        "evaluation_cases_complete": sum(
            bool(status.get("evaluation_complete")) for status in statuses.values()
        ),
        "evaluation_outcomes": outcomes,
        "resolved_instances": resolved_instances,
        "solve_rate": (
            resolved_instances / len(selected_ids)
            if resolved_instances is not None
            else None
        ),
        "detection_metrics": detection_metrics,
        "official_report": official_report,
    }
    atomic_write_json(data_dir / "detection_metrics.json", detection_metrics)
    atomic_write_json(data_dir / f"summary-{run_id}.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    configure_logging(data_dir)
    cases, manifest = prepare_cases(
        data_dir=data_dir,
        sample_size=args.sample_size,
        selection_seed=args.selection_seed,
    )
    selected_ids = list(manifest["instance_ids"])
    LOGGER.info(
        "Selected %d cases from Verified and joined all of them to BM25-13K",
        len(selected_ids),
    )
    if args.prepare_only:
        LOGGER.info("Preparation-only run complete")
        return 0

    settings = GenerationSettings(
        model_id=args.model_id,
        watermarking=args.watermarking,
        prompt_transport=args.prompt_transport,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        gamma=args.gamma if args.watermarking == "wllm" else None,
        delta=args.delta if args.watermarking == "wllm" else None,
        ngram_len=NGRAM_LEN,
        watermark_key=args.watermark_key,
        z_threshold=args.z_threshold,
        generation_seed=args.generation_seed,
    )
    prediction_name = prediction_model_name(settings)
    settings_path = data_dir / "generation_settings.json"
    settings_record = {"schema_version": SCHEMA_VERSION, "settings": asdict(settings)}
    if settings_path.exists():
        existing_settings = json.loads(settings_path.read_text())
        if existing_settings != settings_record:
            raise RuntimeError(
                "Existing generation_settings.json differs from this run; use a clean "
                "data directory instead of mixing resumed generations"
            )
    else:
        atomic_write_json(settings_path, settings_record)
    LOGGER.info(
        "Execution settings: watermarking=%s gamma=%s delta=%s ngram_len=%d "
        "sequential_generation=true max_new_tokens=%d max_input_tokens=%d "
        "asynchronous_modal_workers=%d",
        settings.watermarking,
        settings.gamma,
        settings.delta,
        settings.ngram_len,
        settings.max_new_tokens,
        settings.max_input_tokens,
        args.modal_workers,
    )

    devices = cuda_smoke_test()
    model, tokenizer, device_map = load_qwen(args.model_id, args.allow_model_download)
    logits, detector, watermark_vocab_size = create_watermark_components(
        tokenizer, settings
    )
    prepared_cases = prepare_prompt_cases(tokenizer, cases, settings)
    atomic_write_json(
        data_dir / "prompt_lengths.json",
        {
            "schema_version": SCHEMA_VERSION,
            "order": "selection_manifest_order",
            "cases": [
                {
                    "instance_id": case["instance_id"],
                    "input_tokens": case["_input_tokens"],
                    "selection_index": case["_selection_index"],
                }
                for case in prepared_cases
            ],
        },
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "settings": asdict(settings),
        "selection_file": "selection.json",
        "prediction_model_name": prediction_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "execution": {
            "generation": "sequential",
            "modal_evaluation": "asynchronous_single_case",
            "modal_workers": args.modal_workers,
        },
        "devices": devices,
        "model_device_map": device_map,
        "watermark_vocab_size": watermark_vocab_size,
        "optional_fast_paths": {
            "flash_linear_attention": importlib.util.find_spec("fla") is not None,
            "causal_conv1d": importlib.util.find_spec("causal_conv1d") is not None,
        },
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "accelerate", "datasets", "swebench")
        },
    }
    atomic_write_json(data_dir / f"metadata-{args.run_id}.json", metadata)

    generations_path = data_dir / "generations.jsonl"
    predictions_path = data_dir / "predictions.jsonl"
    detections_path = data_dir / "detections.jsonl"
    completed = read_successful_generations(generations_path)
    write_predictions(predictions_path, selected_ids, completed)
    write_detections(detections_path, selected_ids, completed)

    executor: ThreadPoolExecutor | None = None
    futures: list[Future] = []
    scheduled_ids: set[str] = set()
    if not args.skip_modal:
        executor = ThreadPoolExecutor(
            max_workers=args.modal_workers,
            thread_name_prefix="modal-eval",
        )

    def submit_evaluation(
        generation_index: int, instance_id: str, row: dict[str, Any]
    ) -> None:
        if executor is None or instance_id in scheduled_ids:
            return
        case_dir, evaluation_run_id = prepare_async_evaluation(
            data_dir=data_dir,
            base_run_id=args.run_id,
            generation_index=generation_index,
            instance_id=instance_id,
            row=row,
        )
        future = executor.submit(
            evaluate_one_modal,
            swebench_python=args.swebench_python,
            case_dir=case_dir,
            instance_id=instance_id,
            evaluation_run_id=evaluation_run_id,
            timeout=args.modal_timeout,
        )
        futures.append(future)
        scheduled_ids.add(instance_id)
        LOGGER.info(
            "[%d/%d] Queued asynchronous Modal evaluation for %s",
            generation_index,
            len(prepared_cases),
            instance_id,
        )

    existing_statuses = load_evaluation_statuses(data_dir)
    for generation_index, case in enumerate(prepared_cases, 1):
        instance_id = case["instance_id"]
        status = existing_statuses.get(instance_id)
        already_evaluated = bool(
            status
            and status.get("base_run_id") == args.run_id
            and status.get("evaluation_complete")
        )
        if instance_id in completed and not already_evaluated:
            submit_evaluation(generation_index, instance_id, completed[instance_id])

    generation_failure: str | None = None
    for generation_index, case in enumerate(prepared_cases, 1):
        instance_id = case["instance_id"]
        if instance_id in completed:
            LOGGER.info(
                "[%d/%d] Resume: %s already generated",
                generation_index,
                len(prepared_cases),
                instance_id,
            )
            continue

        LOGGER.info(
            "[%d/%d] Generating %s sequentially (input_tokens=%d)",
            generation_index,
            len(prepared_cases),
            instance_id,
            case["_input_tokens"],
        )
        try:
            row = generate_one(
                model=model,
                tokenizer=tokenizer,
                case=case,
                settings=settings,
                logits=logits,
                detector=detector,
                generation_index=generation_index,
            )
        except Exception as error:
            generation_failure = f"{type(error).__name__}: {error}"
            append_jsonl(
                generations_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "instance_id": instance_id,
                    "generation_index": generation_index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            LOGGER.exception(
                "[%d/%d] Generation failed for %s; draining Modal futures",
                generation_index,
                len(prepared_cases),
                instance_id,
            )
            break

        append_jsonl(generations_path, row)
        completed[instance_id] = row
        write_predictions(predictions_path, selected_ids, completed)
        write_detections(detections_path, selected_ids, completed)
        level = logging.WARNING if row["hit_max_new_tokens"] else logging.INFO
        LOGGER.log(
            level,
            "[%d/%d] Generated %s: tokens=%d stop=%s patch_chars=%d "
            "elapsed=%.1fs throughput=%.2f tokens/s peak_gpu_gib=%s",
            generation_index,
            len(prepared_cases),
            instance_id,
            row["generated_tokens"],
            row["stop_reason"],
            len(row["model_patch"]),
            row["elapsed_seconds"],
            row["tokens_per_second"],
            row["peak_gpu_memory_gib"],
        )
        if row["detection"] is not None:
            if row["detection"]["invalid"]:
                LOGGER.warning(
                    "[%d/%d] WLLM detection %s: invalid completion "
                    "tokens_scored=%s elapsed=%.1fs",
                    generation_index,
                    len(prepared_cases),
                    instance_id,
                    row["detection"]["num_tokens_scored"],
                    row["detection"]["elapsed_seconds"],
                )
            else:
                LOGGER.info(
                    "[%d/%d] WLLM detection %s: z_score=%.4f p_value=%.3e "
                    "green_fraction=%.4f tokens_scored=%d detected=%s "
                    "elapsed=%.1fs",
                    generation_index,
                    len(prepared_cases),
                    instance_id,
                    row["detection"]["z_score"],
                    row["detection"]["p_value"],
                    row["detection"]["green_fraction"],
                    row["detection"]["num_tokens_scored"],
                    row["detection"]["prediction"],
                    row["detection"]["elapsed_seconds"],
                )
        submit_evaluation(generation_index, instance_id, row)

    if not generation_failure:
        missing = sorted(set(selected_ids) - set(completed))
        if missing:
            generation_failure = f"Generation did not finish selected cases: {missing}"
    write_predictions(predictions_path, selected_ids, completed)
    write_detections(detections_path, selected_ids, completed)

    modal_results: list[dict[str, Any]] = []
    modal_future_errors: list[str] = []
    heartbeat: GPUHeartbeat | None = None
    pending_futures = sum(not future.done() for future in futures)
    if pending_futures:
        LOGGER.info(
            "Sequential inference is finished; waiting for %d pending Modal "
            "evaluations before process exit",
            pending_futures,
        )
        heartbeat = GPUHeartbeat(
            data_dir, args.heartbeat_interval, args.heartbeat_busy_seconds
        )
        heartbeat.start()
    try:
        for future in as_completed(futures):
            try:
                modal_results.append(future.result())
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
                modal_future_errors.append(error_text)
                LOGGER.exception(
                    "Unexpected asynchronous Modal future failure; continuing to "
                    "wait for all remaining evaluations"
                )
    finally:
        if heartbeat is not None:
            heartbeat.stop()
        if executor is not None:
            executor.shutdown(wait=True)

    heartbeat_error = heartbeat.error if heartbeat is not None else None
    if heartbeat_error:
        LOGGER.error("GPU heartbeat failed while draining Modal: %s", heartbeat_error)

    official_report = None
    if not args.skip_modal:
        official_report = aggregate_official_reports(
            data_dir=data_dir,
            run_id=args.run_id,
            selected_ids=selected_ids,
            prediction_name=prediction_name,
        )
        log_final_report(official_report)

    summary = build_summary(
        data_dir=data_dir,
        run_id=args.run_id,
        selected_ids=selected_ids,
        generations_path=generations_path,
        official_report=official_report,
        settings=settings,
    )
    summary.update(
        {
            "generation_failure": generation_failure,
            "modal_workers": args.modal_workers,
            "modal_futures_submitted": len(futures),
            "modal_results_collected": len(modal_results),
            "modal_exit_codes": [
                result.get("modal_exit_code") for result in modal_results
            ],
            "modal_future_errors": modal_future_errors,
            "gpu_heartbeat_error": heartbeat_error,
        }
    )
    atomic_write_json(data_dir / f"summary-{args.run_id}.json", summary)
    LOGGER.info(
        "Final detection metrics: %s",
        json.dumps(summary["detection_metrics"], sort_keys=True),
    )
    LOGGER.info("Final summary: %s", json.dumps(summary, sort_keys=True))

    if generation_failure:
        raise RuntimeError(generation_failure)
    if modal_future_errors:
        raise RuntimeError(
            "Unexpected Modal future failures: " + "; ".join(modal_future_errors)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
