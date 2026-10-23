"""Paper-compatible WLLM, SWEET, and SynthID scoring adapters."""

from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import scipy.stats
import torch
from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor, pipeline

from .common import REPO_ROOT


SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _hf_obj import get_synthid_config  # noqa: E402
from _sweet import SweetDetector, WatermarkDetector  # noqa: E402


MODEL_WEIGHT_NAME = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def _cached_snapshot(model_name: str) -> Path | None:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None
    repository = Path(hf_home) / "hub" / ("models--" + model_name.replace("/", "--"))
    reference = repository / "refs" / "main"
    if not reference.is_file():
        return None
    commit = reference.read_text(encoding="utf-8").strip()
    snapshot = repository / "snapshots" / commit
    return snapshot if snapshot.is_dir() else None


@lru_cache(maxsize=None)
def load_tokenizer(model_name: str):
    source: str | Path = _cached_snapshot(model_name) or model_name
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=None)
def load_pipeline(model_name: str):
    weight_name = MODEL_WEIGHT_NAME.get(model_name, model_name)
    source: str | Path = _cached_snapshot(weight_name) or weight_name
    return pipeline(
        "text-generation",
        model=source,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )


def tokenize_segment(text: str, tokenizer) -> torch.Tensor:
    return tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze()


def tokenize_detection(p4d: str, g4d: str, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = tokenize_segment(p4d, tokenizer)
    suffix = tokenize_segment(g4d, tokenizer)
    if len(g4d) == 0 or len(suffix.size()) == 0 or suffix.size(-1) == 0:
        combined = prefix
    else:
        combined = torch.cat((prefix, suffix), dim=0)
    return combined, prefix


def _normal_result(raw: Mapping[str, Any], *, threshold: float = 4.0) -> dict[str, Any]:
    invalid = bool(raw.get("invalid", False))
    z_score = 0.0 if invalid else float(raw["z_score"])
    return {
        "z_score": z_score,
        "p_value": float(scipy.stats.norm.sf(z_score)),
        "invalid": invalid,
        "prediction": bool((not invalid) and z_score > threshold),
        "num_tokens_scored": raw.get("num_tokens_scored"),
        "num_green_tokens": raw.get("num_green_tokens"),
        "green_fraction": raw.get("green_fraction"),
        "watermarking_fraction": raw.get("watermarking_fraction"),
    }


class WllmScorer:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.tokenizer = load_tokenizer(str(config["model_name"]))
        self.detector = WatermarkDetector(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=float(config["gamma"]),
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            ngram_len=int(config["ngram_len"]),
            hash_key=0,
        )

    def score(self, record: Mapping[str, Any], g4d: str) -> dict[str, Any]:
        text, prefix = tokenize_detection(str(record["p4d"]), g4d, self.tokenizer)
        self.detector.hash_key = int(record["custom_seed"])
        raw = self.detector.detect(
            tokenized_text=text,
            tokenized_prefix=prefix,
            return_green_token_mask=False,
        )
        return _normal_result(raw)


class SweetScorer:
    """Exact saved SWEET detector with one cached causal model per process."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.model_name = str(config["model_name"])
        self.tokenizer = load_tokenizer(self.model_name)
        self.pipeline = load_pipeline(self.model_name)
        self.model = self.pipeline.model
        self.detector = SweetDetector(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=float(config["gamma"]),
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            entropy_threshold=float(config["entropy_threshold"]),
            ngram_len=int(config["ngram_len"]),
            hash_key=0,
        )

    def score(self, record: Mapping[str, Any], g4d: str) -> dict[str, Any]:
        text, prefix = tokenize_detection(str(record["p4d"]), g4d, self.tokenizer)
        with torch.no_grad():
            output = self.model(torch.unsqueeze(text, 0).to("cuda"), return_dict=True)
            probabilities = torch.softmax(output.logits, dim=-1)
            entropy = (
                -torch.where(
                    probabilities > 0,
                    probabilities * probabilities.log(),
                    probabilities.new([0.0]),
                ).sum(dim=-1)
            )[0].cpu().tolist()
        self.detector.hash_key = int(record["custom_seed"])
        raw = self.detector.detect(
            tokenized_text=text,
            tokenized_prefix=prefix,
            entropy=entropy,
            return_green_token_mask=False,
        )
        return _normal_result(raw)


class SynthIdScorer:
    def __init__(self, config: Mapping[str, Any], *, device: str = "cpu") -> None:
        self.tokenizer = load_tokenizer(str(config["model_name"]))
        self.ngram_len = int(config["ngram_len"])
        self.device = device

    def score(self, record: Mapping[str, Any], g4d: str) -> dict[str, Any]:
        output_ids = self.tokenizer(g4d, return_tensors="pt").input_ids
        if output_ids.size(-1) < self.ngram_len:
            z_score = 0.0
            count = 0
        else:
            processor = SynthIDTextWatermarkLogitsProcessor(
                device=self.device,
                **get_synthid_config(int(record["custom_seed"]), self.ngram_len),
            )
            observed = processor.compute_g_values(output_ids.to(self.device)).reshape(-1)
            count = int(observed.numel())
            if count == 0:
                z_score = 0.0
            else:
                successes = float(observed.float().sum().item())
                expected = count * 0.5
                z_score = (successes - expected) / math.sqrt(count * 0.25)
        return {
            "z_score": float(z_score),
            "p_value": float(scipy.stats.norm.sf(z_score)),
            "invalid": False,
            "prediction": bool(z_score > 4.0),
            "num_tokens_scored": count,
        }


def make_scorer(config: Mapping[str, Any], *, synthid_device: str = "cpu"):
    watermark = config["watermark"]
    if watermark == "wllm":
        return WllmScorer(config)
    if watermark == "sweet":
        return SweetScorer(config)
    if watermark == "synthid":
        return SynthIdScorer(config, device=synthid_device)
    raise ValueError(f"unsupported watermark: {watermark}")


__all__ = [
    "SweetScorer",
    "SynthIdScorer",
    "WllmScorer",
    "load_tokenizer",
    "make_scorer",
    "tokenize_detection",
]
