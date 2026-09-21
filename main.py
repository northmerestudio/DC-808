#!/usr/bin/env python3
"""Lean identity/character LoRA image curator.

Pipeline:
  1) audit technical signals and capture-time bursts
  2) remove pHash/pixel duplicates
  3) embed whole frames with DINOv2 and remove conservative near-duplicates
  4) tag with a pinned WD14 model and validate every configured tag/prefix
  5) reject clear focus/reliability failures
  6) greedily select train, then validation, using quality + missing coverage + novelty

The source directory is never modified. Output contains:
  OUTPUT/train/
  OUTPUT/validate/
  OUTPUT/debug.csv

Dependencies:
  pip install pillow numpy scipy torch transformers timm huggingface_hub scikit-learn
Optional (used as a soft quality signal only):
  pip install piq
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
from scipy import fft as scipy_fft
from scipy import ndimage as ndi

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif"}


@dataclass
class Record:
    path: str
    width: int = 0
    height: int = 0
    min_side: int = 0
    aspect_ratio: float = 0.0
    capture_time: float = math.nan
    capture_source: str = ""
    burst_id: str = ""
    phash: str = ""
    laplacian: float = math.nan
    exposure: float = math.nan
    blur_global: float = math.nan
    blur_center: float = math.nan
    blur_local: float = math.nan
    prequality: float = 0.0
    focus_risk: float = 0.0
    brisque: float = math.nan
    quality: float = 0.0
    reject_reason: str = ""


@dataclass(frozen=True)
class SemanticSpec:
    name: str
    group: str
    weight: float
    target_fraction: float = 0.0
    source_scale: float = 0.0
    max_target_fraction: float = 1.0
    hard_threshold: float | None = None
    tags: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    exclude_tags: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()


@dataclass
class WD14Vocabulary:
    names: list[str]
    general_indices: np.ndarray
    lookup: dict[str, list[int]]
    resolved_revision: str
    label_path: Path


class BKNode:
    def __init__(self, value: int, index: int):
        self.value = value
        self.index = index
        self.children: dict[int, BKNode] = {}

    def add(self, value: int, index: int) -> None:
        node = self
        while True:
            distance = (node.value ^ value).bit_count()
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value, index)
                return
            node = child

    def find(self, value: int, radius: int) -> list[int]:
        found: list[int] = []
        stack = [self]
        while stack:
            node = stack.pop()
            d = (node.value ^ value).bit_count()
            if d <= radius:
                found.append(node.index)
            lo, hi = d - radius, d + radius
            stack.extend(child for edge, child in node.children.items() if lo <= edge <= hi)
        return found



# ------------------------------ config ---------------------------------------

def _unknown_keys(obj: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise RuntimeError(f"Unknown config keys in {where}: {', '.join(unknown)}")


def _prob(value: Any, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"{name} must be in [0,1]")
    return value


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("Config root must be an object")
    _unknown_keys(cfg, {"dataset", "runtime", "models", "audit", "dedup", "burst", "semantic", "selection", "validation"}, "root")
    required = {"dataset", "runtime", "models", "audit", "dedup", "burst", "semantic", "selection", "validation"}
    missing = sorted(required - set(cfg))
    if missing:
        raise RuntimeError(f"Missing config sections: {', '.join(missing)}")

    ds, rt, models, audit, dedup, burst, sem, sel, val = (
        cfg["dataset"], cfg["runtime"], cfg["models"], cfg["audit"], cfg["dedup"],
        cfg["burst"], cfg["semantic"], cfg["selection"], cfg["validation"]
    )
    _unknown_keys(ds, {"train", "validate"}, "dataset")
    _unknown_keys(rt, {"device", "batch_size", "workers", "output_mode"}, "runtime")
    _unknown_keys(models, {"embedding", "tagger"}, "models")
    _unknown_keys(models["tagger"], {"repo", "revision", "selected_tags_sha256"}, "models.tagger")
    _unknown_keys(audit, {"min_side", "max_aspect", "brisque", "focus"}, "audit")
    _unknown_keys(audit["brisque"], {"enabled", "weight"}, "audit.brisque")
    _unknown_keys(audit["focus"], {"fail_risk", "soft_risk_start", "semantic_hard", "semantic_corroborated", "semantic_corroboration_risk"}, "audit.focus")
    _unknown_keys(dedup, {"phash_distance", "pixel_difference", "dino_similarity", "dino_phash_distance"}, "dedup")
    _unknown_keys(burst, {"enabled", "gap_seconds", "max_span_seconds", "use_exif", "use_filename", "use_mtime", "repeat_penalty"}, "burst")
    _unknown_keys(sem, {"evidence_floor", "presence_threshold", "availability_capture", "categories", "penalties"}, "semantic")
    _unknown_keys(sel, {"quality_weight", "coverage_weight", "novelty_weight", "coverage_top_groups", "novelty_similarity_start", "redundancy_similarity_start", "redundancy_penalty", "soft_focus_penalty", "semantic_penalty_cap"}, "selection")
    _unknown_keys(val, {"near_duplicate_similarity", "near_duplicate_phash_distance", "same_burst_similarity"}, "validation")

    if int(ds["train"]) < 1 or int(ds["validate"]) < 0:
        raise RuntimeError("dataset.train must be >0 and dataset.validate >=0")
    if int(rt["batch_size"]) < 1 or int(rt["workers"]) < 1:
        raise RuntimeError("runtime batch_size/workers must be positive")
    if rt["device"] not in {"auto", "cpu", "cuda", "mps"}:
        raise RuntimeError("runtime.device must be auto/cpu/cuda/mps")
    if rt["output_mode"] not in {"copy", "hardlink", "symlink"}:
        raise RuntimeError("runtime.output_mode must be copy/hardlink/symlink")
    if not 0.0 <= float(audit["brisque"]["weight"]) <= 1.0:
        raise RuntimeError("audit.brisque.weight must be in [0,1]")
    if int(audit["min_side"]) < 1 or float(audit["max_aspect"]) < 1.0:
        raise RuntimeError("invalid audit dimensions")
    f = audit["focus"]
    for k in f:
        _prob(f[k], f"audit.focus.{k}")
    if not float(f["soft_risk_start"]) < float(f["fail_risk"]):
        raise RuntimeError("focus.soft_risk_start must be below focus.fail_risk")
    if not 0 <= int(dedup["phash_distance"]) <= 64 or not 0 <= int(dedup["dino_phash_distance"]) <= 64:
        raise RuntimeError("pHash distances must be in [0,64]")
    _prob(dedup["pixel_difference"], "dedup.pixel_difference")
    _prob(dedup["dino_similarity"], "dedup.dino_similarity")
    if float(burst["gap_seconds"]) < 0 or float(burst["max_span_seconds"]) < float(burst["gap_seconds"]):
        raise RuntimeError("burst times must be nonnegative and max_span_seconds >= gap_seconds")
    if not 0.0 <= float(burst["repeat_penalty"]) <= 1.0:
        raise RuntimeError("burst.repeat_penalty must be in [0,1]")
    floor = _prob(sem["evidence_floor"], "semantic.evidence_floor")
    presence = _prob(sem["presence_threshold"], "semantic.presence_threshold")
    _prob(sem["availability_capture"], "semantic.availability_capture")
    if not floor < presence:
        raise RuntimeError("semantic.evidence_floor must be below semantic.presence_threshold")
    weights = [float(sel[k]) for k in ("quality_weight", "coverage_weight", "novelty_weight")]
    if any(x < 0 for x in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise RuntimeError("selection quality/coverage/novelty weights must be nonnegative and sum to 1")
    if int(sel["coverage_top_groups"]) < 1:
        raise RuntimeError("selection.coverage_top_groups must be positive")
    for k in ("novelty_similarity_start", "redundancy_similarity_start"):
        _prob(sel[k], f"selection.{k}")
    for k in ("redundancy_penalty", "soft_focus_penalty", "semantic_penalty_cap"):
        if not 0.0 <= float(sel[k]) <= 1.0:
            raise RuntimeError(f"selection.{k} must be in [0,1]")
    _prob(val["near_duplicate_similarity"], "validation.near_duplicate_similarity")
    _prob(val["same_burst_similarity"], "validation.same_burst_similarity")
    if not 0 <= int(val["near_duplicate_phash_distance"]) <= 64:
        raise RuntimeError("validation.near_duplicate_phash_distance must be in [0,64]")

    names: set[str] = set()
    allowed_spec = {"name", "group", "weight", "target_fraction", "source_scale", "max_target_fraction", "hard_threshold", "tags", "prefixes", "exclude_tags", "exclude_prefixes"}
    for block in ("categories", "penalties"):
        if not isinstance(sem[block], list):
            raise RuntimeError(f"semantic.{block} must be a list")
        for raw in sem[block]:
            if not isinstance(raw, dict):
                raise RuntimeError(f"semantic.{block} entries must be objects")
            _unknown_keys(raw, allowed_spec, f"semantic.{block}.{raw.get('name','?')}")
            name = str(raw.get("name", "")).strip()
            group = str(raw.get("group", block)).strip()
            if not name or not group or name in names:
                raise RuntimeError(f"Invalid/duplicate semantic spec name: {name!r}")
            names.add(name)
            if float(raw.get("weight", 0.0)) < 0:
                raise RuntimeError(f"{name}.weight must be nonnegative")
            for key in ("target_fraction", "source_scale", "max_target_fraction", "hard_threshold"):
                if key in raw and raw[key] is not None:
                    _prob(raw[key], f"{name}.{key}")
            if float(raw.get("max_target_fraction", 1.0)) < float(raw.get("target_fraction", 0.0)):
                raise RuntimeError(f"{name}.max_target_fraction cannot be below target_fraction")
            if not raw.get("tags") and not raw.get("prefixes"):
                raise RuntimeError(f"{name} needs tags and/or prefixes")
    if "semantic_blur" not in names:
        raise RuntimeError("semantic.penalties must include semantic_blur")
    return cfg


def semantic_specs(cfg: Mapping[str, Any]) -> tuple[list[SemanticSpec], list[SemanticSpec]]:
    def build(raw: Mapping[str, Any], default_group: str) -> SemanticSpec:
        threshold = raw.get("hard_threshold")
        return SemanticSpec(
            name=str(raw["name"]),
            group=str(raw.get("group", default_group)),
            weight=float(raw.get("weight", 0.0)),
            target_fraction=float(raw.get("target_fraction", 0.0)),
            source_scale=float(raw.get("source_scale", 0.0)),
            max_target_fraction=float(raw.get("max_target_fraction", 1.0)),
            hard_threshold=None if threshold is None else float(threshold),
            tags=tuple(map(str, raw.get("tags", ()))),
            prefixes=tuple(map(str, raw.get("prefixes", ()))),
            exclude_tags=tuple(map(str, raw.get("exclude_tags", ()))),
            exclude_prefixes=tuple(map(str, raw.get("exclude_prefixes", ()))),
        )

    categories = [build(item, "coverage") for item in cfg["semantic"]["categories"]]
    penalties = [build(item, "penalty") for item in cfg["semantic"]["penalties"]]
    return categories, penalties


# ------------------------------ images ---------------------------------------

def image_paths(root: Path, output: Path) -> list[Path]:
    root, output = root.resolve(), output.resolve()
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            rp = p.resolve()
            if output == rp or output in rp.parents:
                continue
            out.append(rp)
    return sorted(out)


def _rgb_from_loaded(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    if has_alpha:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return image.convert("RGB")


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return _rgb_from_loaded(im)


def pad_square(image: Image.Image, neutral: bool) -> Image.Image:
    if image.width == image.height:
        return image
    side = max(image.width, image.height)
    fill = image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0)) if neutral else (255, 255, 255)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def robust_rank(values: Sequence[float], higher_is_better: bool = True) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(a)
    out = np.full(len(a), 0.5, dtype=np.float32)
    if finite.sum() <= 1:
        return out
    vals = a[finite]
    _, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    if len(counts) == 1:
        return out
    cumulative = np.cumsum(counts)
    average = (cumulative - counts + cumulative - 1) / 2.0
    ranks = (average[inv] / (len(vals) - 1)).astype(np.float32)
    out[finite] = ranks if higher_is_better else 1.0 - ranks
    return out


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> str:
    side = hash_size * 4
    gray = image.convert("L").resize((side, side), Image.Resampling.LANCZOS)
    coeff = scipy_fft.dctn(np.asarray(gray, np.float32), type=2, norm="ortho")[:hash_size, :hash_size].copy()
    median = float(np.median(coeff.ravel()[1:]))
    bits = coeff > median
    bits[0, 0] = False
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def phash_distance(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def simple_signals(image: Image.Image) -> tuple[float, float]:
    img = image.copy()
    if max(img.size) > 768:
        scale = 768 / max(img.size)
        img = img.resize(tuple(max(32, round(v * scale)) for v in img.size), Image.Resampling.LANCZOS)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    lap = float(ndi.laplace(gray).var())
    clipped = float(np.mean((gray <= 4) | (gray >= 251)))
    median = float(np.median(gray))
    if median < 48:
        brightness = float(np.clip((median - 8) / 40, 0, 1))
    elif median > 207:
        brightness = float(np.clip((247 - median) / 40, 0, 1))
    else:
        brightness = 1.0
    p05, p95 = np.percentile(gray, (5, 95))
    spread = float(np.clip((p95 - p05) / 160, 0, 1))
    exposure = float(np.clip((1 - clipped) * (0.75 * brightness + 0.25 * spread), 0, 1))
    return lap, exposure


def _reblur(gray: np.ndarray, size: int) -> float:
    x = np.asarray(gray, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 16:
        return math.nan
    if float(np.nanmax(x)) > 1.5:
        x = x / 255.0
    core = tuple(slice(2, n - 1) for n in x.shape)
    scores: list[float] = []
    eps = 1e-12
    for axis in (0, 1):
        rb = ndi.uniform_filter1d(x, size, axis=axis, mode="reflect")
        e0 = np.abs(ndi.sobel(x, axis=axis, mode="reflect"))
        e1 = np.abs(ndi.sobel(rb, axis=axis, mode="reflect"))
        e0 = np.maximum(e0, eps)
        lost = np.maximum(0.0, e0 - np.maximum(e1, eps))
        total = float(np.sum(e0[core])); lost_sum = float(np.sum(lost[core]))
        scores.append(1.0 if total <= eps else float(np.clip(abs(total - lost_sum) / total, 0, 1)))
    return max(scores)


def blur_signals(image: Image.Image) -> tuple[float, float, float]:
    img = image.convert("L")
    if max(img.size) > 384:
        scale = 384 / max(img.size)
        img = img.resize(tuple(max(32, round(v * scale)) for v in img.size), Image.Resampling.LANCZOS)
    gray = np.asarray(img, dtype=np.float32) / 255.0
    global_blur = _reblur(gray, 11)
    h, w = gray.shape
    center = gray[int(.16*h):int(.84*h), int(.16*w):int(.84*w)]
    center_blur = _reblur(center, 9)
    bs, ts = [], []
    for r in range(4):
        for c in range(4):
            tile = gray[r*h//4:(r+1)*h//4, c*w//4:(c+1)*w//4]
            if min(tile.shape) < 16:
                continue
            bs.append(_reblur(tile, 7))
            gx, gy = ndi.sobel(tile, axis=1, mode="reflect"), ndi.sobel(tile, axis=0, mode="reflect")
            ts.append(float(np.mean(np.hypot(gx, gy))))
    if not bs:
        return global_blur, center_blur, global_blur
    ba, ta = np.asarray(bs), np.asarray(ts)
    floor = max(.006, float(np.quantile(ta, .25)))
    vals = ba[ta >= floor]
    if not len(vals): vals = ba
    return global_blur, center_blur, float(np.quantile(vals, .90))


_EXIF_DT_ORIGINAL, _EXIF_DT_DIGITIZED, _EXIF_DT = 36867, 36868, 306
_EXIF_SUB_ORIGINAL, _EXIF_SUB_DIGITIZED = 37521, 37522
_FILENAME_SUB = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[^\d]?(\d{2})(\d{2})(\d{2})(\d{3,6})(?!\d)")
_FILENAME_TIMES = (
    re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[^\d]?(\d{2})(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})[ T_-]+(\d{2})[-_.:](\d{2})[-_.:](\d{2})(?!\d)"),
)


def _timestamp(value: object, sub: object = None) -> float | None:
    if value is None: return None
    text = str(value).strip().replace("-", ":", 2)
    parsed = None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z"):
        try:
            parsed = datetime.strptime(text, fmt); break
        except ValueError: pass
    if parsed is None: return None
    if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=timezone.utc)
    frac = 0.0
    if sub is not None:
        digits = re.sub(r"\D", "", str(sub))[:6]
        if digits: frac = int(digits) / (10 ** len(digits))
    return parsed.timestamp() + frac


def capture_time(path: Path, image: Image.Image, cfg: Mapping[str, Any]) -> tuple[float, str]:
    b = cfg["burst"]
    if not b["enabled"]: return math.nan, ""
    if b["use_exif"]:
        try:
            exif = image.getexif()
            t = _timestamp(exif.get(_EXIF_DT_ORIGINAL), exif.get(_EXIF_SUB_ORIGINAL))
            if t is None: t = _timestamp(exif.get(_EXIF_DT_DIGITIZED), exif.get(_EXIF_SUB_DIGITIZED))
            if t is None: t = _timestamp(exif.get(_EXIF_DT))
            if t is not None: return t, "exif"
        except Exception: pass
    if b["use_filename"]:
        m = _FILENAME_SUB.search(path.stem)
        if m:
            try:
                vals = [int(x) for x in m.groups()[:6]]; digits = m.group(7)
                return datetime(*vals, tzinfo=timezone.utc).timestamp() + int(digits)/(10**len(digits)), "filename"
            except ValueError: pass
        for pat in _FILENAME_TIMES:
            m = pat.search(path.stem)
            if m:
                try: return datetime(*[int(x) for x in m.groups()], tzinfo=timezone.utc).timestamp(), "filename"
                except ValueError: pass
    if b["use_mtime"]:
        try: return path.stat().st_mtime, "mtime"
        except OSError: pass
    return math.nan, ""


def audit_one(path: Path, cfg: Mapping[str, Any]) -> Record:
    r = Record(path=str(path))
    try:
        with Image.open(path) as im:
            im.load()
            r.capture_time, r.capture_source = capture_time(path, im, cfg)
            rgb = _rgb_from_loaded(im)
        r.width, r.height = rgb.size
        r.min_side = min(rgb.size)
        r.aspect_ratio = max(rgb.size) / max(1, min(rgb.size))
        if r.min_side < int(cfg["audit"]["min_side"]):
            r.reject_reason = "small"
            return r
        if r.aspect_ratio > float(cfg["audit"]["max_aspect"]):
            r.reject_reason = "aspect"
            return r
        r.phash = perceptual_hash(rgb)
        r.laplacian, r.exposure = simple_signals(rgb)
        r.blur_global, r.blur_center, r.blur_local = blur_signals(rgb)
    except Exception as exc:
        r.reject_reason = f"decode:{type(exc).__name__}"
    return r


def assign_bursts(records: list[Record], cfg: Mapping[str, Any]) -> None:
    b = cfg["burst"]
    if not b["enabled"]: return
    buckets: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(records):
        if math.isfinite(r.capture_time):
            buckets.setdefault((str(Path(r.path).parent), r.capture_source), []).append(i)
    number = 0
    for key in sorted(buckets):
        ids = sorted(buckets[key], key=lambda i: (records[i].capture_time, records[i].path))
        group: list[int] = []; start = prev = math.nan
        def commit(g: list[int]) -> None:
            nonlocal number
            if len(g) >= 2:
                number += 1
                bid = f"B{number:05d}"
                for j in g: records[j].burst_id = bid
        for i in ids:
            t = records[i].capture_time
            if not group:
                group = [i]; start = prev = t; continue
            if t - prev <= float(b["gap_seconds"]) and t - start <= float(b["max_span_seconds"]):
                group.append(i); prev = t
            else:
                commit(group); group = [i]; start = prev = t
        commit(group)


# ------------------------------ dedup ----------------------------------------

def resolution_score(r: Record, min_side: int) -> float:
    return float(np.clip(math.log2(max(1.0, r.min_side / min_side)) / 2.0, 0.0, 1.0))


def assign_prequality(records: list[Record], ids: Sequence[int], cfg: Mapping[str, Any]) -> None:
    sharpness_rank = robust_rank([records[i].laplacian for i in ids], True)
    min_side = int(cfg["audit"]["min_side"])

    for pos, i in enumerate(ids):
        record = records[i]
        resolution = resolution_score(record, min_side)
        record.prequality = float(
            0.65 * sharpness_rank[pos]
            + 0.25 * record.exposure
            + 0.10 * resolution
        )


@lru_cache(maxsize=8192)
def _thumb(path: str) -> np.ndarray:
    img = open_rgb(Path(path)).resize((128, 128), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def pixel_difference(a: str, b: str) -> float:
    return float(np.mean(np.abs(_thumb(a) - _thumb(b))) / 255.0)


def phash_dedup(records: list[Record], ids: Sequence[int], cfg: Mapping[str, Any]) -> list[int]:
    radius = int(cfg["dedup"]["phash_distance"]); maxdiff = float(cfg["dedup"]["pixel_difference"])
    ordered = sorted(ids, key=lambda i: (-records[i].prequality, records[i].path))
    kept: list[int] = []; tree: BKNode | None = None
    for i in ordered:
        value = int(records[i].phash, 16)
        matches = [] if tree is None else tree.find(value, radius)
        duplicate = next((j for j in matches if pixel_difference(records[i].path, records[j].path) <= maxdiff), None)
        if duplicate is not None:
            records[i].reject_reason = "phash_duplicate"
            continue
        kept.append(i)
        if tree is None: tree = BKNode(value, i)
        else: tree.add(value, i)
    return sorted(kept, key=lambda i: records[i].path)


def resolve_device(requested: str) -> str:
    import torch
    if requested != "auto": return requested
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def dino_embeddings(paths: Sequence[str], model_name: str, device: str, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    result: np.ndarray | None = None
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start+batch_size]
        images = [pad_square(open_rgb(Path(p)), True).resize((224, 224), Image.Resampling.LANCZOS) for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", do_resize=False, do_center_crop=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs)
            feat = getattr(out, "pooler_output", None)
            if feat is None: feat = out.last_hidden_state[:, 0]
            feat = torch.nn.functional.normalize(feat.float(), dim=1)
        arr = feat.cpu().numpy().astype(np.float32, copy=False)
        if result is None: result = np.empty((len(paths), arr.shape[1]), np.float32)
        result[start:start+len(arr)] = arr
        print(f"  DINO {min(start+batch_size,len(paths))}/{len(paths)}", file=sys.stderr)
    del model
    if device == "cuda": torch.cuda.empty_cache()
    return np.empty((0, 0), np.float32) if result is None else result


def semantic_dedup(records: list[Record], ids: Sequence[int], emb: np.ndarray, cfg: Mapping[str, Any]) -> tuple[list[int], np.ndarray]:
    """Best-first all-neighbor semantic dedup without transitive chaining."""
    if len(ids) < 2:
        return list(ids), emb
    from sklearn.neighbors import NearestNeighbors
    threshold = float(cfg["dedup"]["dino_similarity"])
    ph = int(cfg["dedup"]["dino_phash_distance"])
    nn = NearestNeighbors(
        radius=max(0.0, 1.0 - threshold + 1e-7),
        metric="cosine", algorithm="brute", n_jobs=-1,
    ).fit(emb)
    distances, neighbors = nn.radius_neighbors(emb, return_distance=True)
    ordered = sorted(
        range(len(ids)),
        key=lambda pos: (-records[ids[pos]].prequality, records[ids[pos]].path),
    )
    kept_positions: set[int] = set()
    keep_pos: list[int] = []

    for pos in ordered:
        duplicate = False
        current = records[ids[pos]]

        for distance, neighbor_raw in zip(distances[pos], neighbors[pos]):
            neighbor = int(neighbor_raw)
            if neighbor == pos or neighbor not in kept_positions:
                continue

            similarity = 1.0 - float(distance)
            if similarity < threshold:
                continue

            kept_record = records[ids[neighbor]]
            if phash_distance(current.phash, kept_record.phash) <= ph:
                duplicate = True
                break

        if duplicate:
            current.reject_reason = "dino_duplicate"
        else:
            kept_positions.add(pos)
            keep_pos.append(pos)
    keep_pos.sort()
    return [ids[p] for p in keep_pos], emb[keep_pos]


# ------------------------------ WD14 -----------------------------------------

def norm_tag(x: str) -> str:
    return re.sub(r"\s+", "_", x.strip().lower())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def load_wd14_vocab(cfg: Mapping[str, Any]) -> WD14Vocabulary:
    from huggingface_hub import hf_hub_download
    t = cfg["models"]["tagger"]
    path = Path(hf_hub_download(t["repo"], "selected_tags.csv", revision=t["revision"]))
    expected = str(t.get("selected_tags_sha256", "")).lower()
    actual = sha256_file(path).lower()
    if expected and expected != actual:
        raise RuntimeError(f"WD14 selected_tags.csv hash mismatch: expected {expected}, got {actual}")
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    names = [norm_tag(r["name"]) for r in rows]
    categories = [int(r.get("category", 0)) for r in rows]
    general = np.asarray([i for i, c in enumerate(categories) if c == 0], np.int64)
    lookup: dict[str, list[int]] = {}
    for i in general.tolist(): lookup.setdefault(names[i], []).append(i)
    resolved = ""
    parts = list(path.parts)
    if "snapshots" in parts:
        k = parts.index("snapshots")
        if k+1 < len(parts): resolved = parts[k+1]
    return WD14Vocabulary(names, general, lookup, resolved, path)


def resolve_semantics(vocab: WD14Vocabulary, specs: Sequence[SemanticSpec]) -> dict[str, np.ndarray]:
    general = vocab.general_indices.tolist(); names = vocab.names; errors: list[str] = []; result: dict[str, np.ndarray] = {}
    def prefix(p: str) -> list[int]:
        p = norm_tag(p); return [i for i in general if names[i].startswith(p)]
    for s in specs:
        matched: set[int] = set()
        for tag in s.tags:
            found = vocab.lookup.get(norm_tag(tag), [])
            if not found: errors.append(f"{s.name}.tags:{tag}")
            matched.update(found)
        for p in s.prefixes:
            found = prefix(p)
            if not found: errors.append(f"{s.name}.prefixes:{p}*")
            matched.update(found)
        excluded: set[int] = set()
        for tag in s.exclude_tags:
            found = vocab.lookup.get(norm_tag(tag), [])
            if not found: errors.append(f"{s.name}.exclude_tags:{tag}")
            excluded.update(found)
        for p in s.exclude_prefixes:
            found = prefix(p)
            if not found: errors.append(f"{s.name}.exclude_prefixes:{p}*")
            excluded.update(found)
        matched.difference_update(excluded)
        if not matched: errors.append(f"{s.name}:matched_nothing")
        result[s.name] = np.asarray(sorted(matched), np.int64)
    if errors:
        raise RuntimeError("WD14 vocabulary contract failed: " + ", ".join(errors[:30]) + (" ..." if len(errors)>30 else ""))
    return result


def wd14_scores(records: Sequence[Record], ids: Sequence[int], cfg: Mapping[str, Any], specs: Sequence[SemanticSpec], vocab: WD14Vocabulary, indexes: Mapping[str, np.ndarray], device: str) -> np.ndarray:
    import torch, timm
    repo = cfg["models"]["tagger"]["repo"]
    revision = vocab.resolved_revision or cfg["models"]["tagger"]["revision"]
    model = timm.create_model(f"hf_hub:{repo}@{revision}", pretrained=True).to(device).eval()
    dc = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**dc, is_training=False)
    out = np.zeros((len(ids), len(specs)), np.float32)
    batch_size = int(cfg["runtime"]["batch_size"])
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start+batch_size]
        images = [pad_square(open_rgb(Path(records[i].path)), False) for i in chunk]
        tensor = torch.stack([transform(im) for im in images])[:, [2,1,0], :, :].to(device)
        with torch.inference_mode():
            logits = model(tensor)
            if isinstance(logits, (tuple, list)): logits = logits[0]
            probs = torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32)
        if probs.shape[1] != len(vocab.names):
            raise RuntimeError(f"WD14 output width {probs.shape[1]} != vocabulary {len(vocab.names)}")
        for j, s in enumerate(specs): out[start:start+len(chunk), j] = probs[:, indexes[s.name]].max(axis=1)
        print(f"  WD14 {min(start+batch_size,len(ids))}/{len(ids)}", file=sys.stderr)
    del model
    if device == "cuda": torch.cuda.empty_cache()
    return out


# ------------------------------ eligibility/quality --------------------------

def focus_assessment(records: Sequence[Record], ids: Sequence[int], semantic_blur: np.ndarray, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    gr = robust_rank([records[i].blur_global for i in ids], True)
    cr = robust_rank([records[i].blur_center for i in ids], True)
    lr = robust_rank([records[i].blur_local for i in ids], True)
    sharp = robust_rank([records[i].laplacian for i in ids], True)
    exposure = np.asarray([records[i].exposure for i in ids], np.float32)
    consensus = np.partition(np.stack([gr, cr, lr], axis=1), 1, axis=1)[:, 1]
    risk = np.clip(.50*consensus + .40*(1.0-sharp) + .10*(1.0-exposure), 0, 1).astype(np.float32)
    f = cfg["audit"]["focus"]
    hard = ((risk >= float(f["fail_risk"])) |
            (semantic_blur >= float(f["semantic_hard"])) |
            ((semantic_blur >= float(f["semantic_corroborated"])) & (risk >= float(f["semantic_corroboration_risk"]))))
    for p, i in enumerate(ids): records[i].focus_risk = float(risk[p])
    return risk, hard


def compute_brisque(records: list[Record], ids: Sequence[int], device: str, enabled: bool) -> bool:
    if not enabled or not ids: return False
    try:
        import torch, piq
    except ImportError:
        print("Warning: piq unavailable; BRISQUE disabled (soft quality only).", file=sys.stderr)
        return False
    for n, i in enumerate(ids, 1):
        try:
            img = open_rgb(Path(records[i].path))
            if max(img.size) > 768:
                scale = 768/max(img.size); img = img.resize(tuple(max(32, round(v*scale)) for v in img.size), Image.Resampling.LANCZOS)
            arr = np.asarray(img, np.float32)/255.0
            tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)
            with torch.inference_mode(): records[i].brisque = float(piq.brisque(tensor, data_range=1.0).item())
        except Exception: records[i].brisque = math.nan
        if n % 100 == 0 or n == len(ids): print(f"  BRISQUE {n}/{len(ids)}", file=sys.stderr)
    return sum(math.isfinite(records[i].brisque) for i in ids) >= 2


def assign_quality(records: list[Record], ids: Sequence[int], cfg: Mapping[str, Any], brisque_ok: bool) -> None:
    focus_q = np.asarray([1.0-records[i].focus_risk for i in ids], np.float32)
    exposure = np.asarray([records[i].exposure for i in ids], np.float32)
    resolution = np.asarray([resolution_score(records[i], int(cfg["audit"]["min_side"])) for i in ids], np.float32)
    pieces = [(focus_q, .55), (exposure, .20), (resolution, .10)]
    if brisque_ok:
        pieces.append((robust_rank([records[i].brisque for i in ids], False), float(cfg["audit"]["brisque"]["weight"])))
    total_w = sum(w for _, w in pieces)
    q = sum(arr*w for arr, w in pieces) / total_w
    for p, i in enumerate(ids): records[i].quality = float(q[p])


def evidence_transform(raw: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    floor = float(cfg["semantic"]["evidence_floor"])
    presence = float(cfg["semantic"]["presence_threshold"])
    return np.clip((raw - floor) / (presence - floor), 0.0, 1.0).astype(np.float32)


# ------------------------------ selection ------------------------------------

def coverage_targets(raw: np.ndarray, evidence: np.ndarray, specs: Sequence[SemanticSpec], target_n: int, cfg: Mapping[str, Any]) -> np.ndarray:
    threshold = float(cfg["semantic"]["presence_threshold"]); capture = float(cfg["semantic"]["availability_capture"])
    targets = np.zeros(len(specs), np.float32)
    for j, s in enumerate(specs):
        prevalence = float(np.mean(raw[:, j] >= threshold)) if len(raw) else 0.0
        requested_frac = max(s.target_fraction, s.source_scale * prevalence)
        requested_frac = min(requested_frac, s.max_target_fraction)
        requested = requested_frac * target_n
        available = float(np.sum(raw[:, j] >= threshold)) * capture
        targets[j] = min(requested, available)
    return targets


def semantic_penalties(raw_penalty: np.ndarray, specs: Sequence[SemanticSpec], cfg: Mapping[str, Any]) -> np.ndarray:
    ev = evidence_transform(raw_penalty, cfg)
    weights = np.asarray([s.weight for s in specs], np.float32)
    cap = float(cfg["selection"]["semantic_penalty_cap"])
    return np.clip(ev @ weights, 0.0, cap).astype(np.float32)


def hard_semantic_reject(raw_penalty: np.ndarray, specs: Sequence[SemanticSpec]) -> np.ndarray:
    hard = np.zeros(len(raw_penalty), dtype=bool)
    for j, s in enumerate(specs):
        if s.hard_threshold is not None: hard |= raw_penalty[:, j] >= s.hard_threshold
    return hard


def _coverage_scores(candidates: np.ndarray, evidence: np.ndarray, coverage: np.ndarray, targets: np.ndarray, specs: Sequence[SemanticSpec], top_k: int) -> np.ndarray:
    deficits = np.divide(np.maximum(targets-coverage, 0.0), np.maximum(targets, 1e-6), out=np.zeros_like(targets), where=targets>1e-6)
    weights = np.asarray([s.weight for s in specs], np.float32)
    max_weight = max(1e-6, float(weights.max(initial=1.0)))
    gain = evidence[candidates] * deficits[None, :] * (weights[None, :] / max_weight)
    group_columns: dict[str, list[int]] = {}
    for j, spec in enumerate(specs):
        group_columns.setdefault(spec.group, []).append(j)

    grouped = np.zeros((len(candidates), len(group_columns)), np.float32)
    for group_index, cols in enumerate(group_columns.values()):
        grouped[:, group_index] = gain[:, cols].max(axis=1)
    k = min(top_k, grouped.shape[1])
    if k <= 0: return np.zeros(len(candidates), np.float32)
    top = np.partition(grouped, grouped.shape[1]-k, axis=1)[:, -k:]
    counts = np.maximum(1, np.sum(top > 1e-6, axis=1))
    return (np.sum(top, axis=1) / counts).astype(np.float32)


def select_greedy(
    records: Sequence[Record],
    pool_positions: Sequence[int],
    embeddings: np.ndarray,
    raw_categories: np.ndarray,
    category_specs: Sequence[SemanticSpec],
    penalty_scores: np.ndarray,
    target_n: int,
    cfg: Mapping[str, Any],
    initial_similarity: np.ndarray | None = None,
    initial_burst_counts: Mapping[str, int] | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    positions = np.asarray(pool_positions, np.int64)
    if not len(positions) or target_n <= 0:
        empty = np.zeros(len(category_specs), np.float32)
        return [], empty.copy(), empty

    evidence = evidence_transform(raw_categories, cfg)
    targets = coverage_targets(
        raw_categories[positions],
        evidence[positions],
        category_specs,
        target_n,
        cfg,
    )
    coverage = np.zeros(len(category_specs), np.float32)

    quality = np.asarray([record.quality for record in records], np.float32)
    focus = np.asarray([record.focus_risk for record in records], np.float32)

    selected: list[int] = []
    remaining = positions.tolist()
    max_similarity = (
        np.full(len(records), -1.0, np.float32)
        if initial_similarity is None
        else initial_similarity.astype(np.float32, copy=True)
    )
    burst_counts = dict(initial_burst_counts or {})

    selection_cfg = cfg["selection"]
    burst_cfg = cfg["burst"]
    focus_cfg = cfg["audit"]["focus"]

    novelty_start = float(selection_cfg["novelty_similarity_start"])
    redundancy_start = float(selection_cfg["redundancy_similarity_start"])
    redundancy_weight = float(selection_cfg["redundancy_penalty"])
    soft_focus_weight = float(selection_cfg["soft_focus_penalty"])
    burst_weight = float(burst_cfg["repeat_penalty"])

    quality_weight = float(selection_cfg["quality_weight"])
    coverage_weight = float(selection_cfg["coverage_weight"])
    novelty_weight = float(selection_cfg["novelty_weight"])

    soft_focus_start = float(focus_cfg["soft_risk_start"])
    hard_focus = float(focus_cfg["fail_risk"])
    coverage_top_groups = int(selection_cfg["coverage_top_groups"])

    for _ in range(min(target_n, len(remaining))):
        candidates = np.asarray(remaining, np.int64)
        candidate_similarity = max_similarity[candidates]

        if selected or np.any(candidate_similarity > -0.5):
            novelty = np.clip(
                (1.0 - candidate_similarity) / (1.0 - novelty_start),
                0.0,
                1.0,
            )
        else:
            novelty = np.full(len(candidates), 0.5, np.float32)

        coverage_score = _coverage_scores(
            candidates,
            evidence,
            coverage,
            targets,
            category_specs,
            coverage_top_groups,
        )

        redundancy_proximity = np.clip(
            (candidate_similarity - redundancy_start) / max(1e-6, 1.0 - redundancy_start),
            0.0,
            1.0,
        )
        redundancy_penalty = redundancy_weight * redundancy_proximity**2

        soft_focus = np.clip(
            (focus[candidates] - soft_focus_start) / max(1e-6, hard_focus - soft_focus_start),
            0.0,
            1.0,
        )
        focus_penalty = soft_focus_weight * soft_focus

        burst_penalty = np.asarray(
            [
                burst_weight * min(3, burst_counts.get(records[i].burst_id, 0))
                if records[i].burst_id
                else 0.0
                for i in candidates
            ],
            np.float32,
        )

        total_score = (
            quality_weight * quality[candidates]
            + coverage_weight * coverage_score
            + novelty_weight * novelty
            - penalty_scores[candidates]
            - redundancy_penalty
            - focus_penalty
            - burst_penalty
        )

        best_score = float(total_score.max())
        ties = np.flatnonzero(np.isclose(total_score, best_score, rtol=0.0, atol=1e-8))
        best_local = int(ties[np.argmax(quality[candidates][ties])])
        winner = int(candidates[best_local])

        selected.append(winner)
        remaining.remove(winner)
        coverage += evidence[winner]

        similarities = embeddings @ embeddings[winner]
        max_similarity = np.maximum(max_similarity, similarities.astype(np.float32))

        burst_id = records[winner].burst_id
        if burst_id:
            burst_counts[burst_id] = burst_counts.get(burst_id, 0) + 1

    return selected, coverage, targets


def validation_pool(records: Sequence[Record], eligible: Sequence[int], training: Sequence[int], embeddings: np.ndarray, cfg: Mapping[str, Any]) -> tuple[list[int], np.ndarray]:
    train = np.asarray(training, np.int64)
    if not len(train): return list(eligible), np.full(len(records), -1.0, np.float32)
    max_sim = np.full(len(records), -1.0, np.float32)
    training_set = set(training)
    candidates = [i for i in eligible if i not in training_set]
    if not candidates: return [], max_sim
    train_matrix = embeddings[train]
    cand_arr = np.asarray(candidates, np.int64)
    sims = embeddings[cand_arr] @ train_matrix.T
    max_sim[cand_arr] = sims.max(axis=1)
    keep: list[int] = []
    sim_thr = float(cfg["validation"]["near_duplicate_similarity"]); ph_thr = int(cfg["validation"]["near_duplicate_phash_distance"]); burst_thr = float(cfg["validation"]["same_burst_similarity"])
    train_by_burst: dict[str, list[int]] = {}
    for t in training:
        if records[t].burst_id: train_by_burst.setdefault(records[t].burst_id, []).append(t)
    for row, c in enumerate(candidates):
        bad = False
        strong = np.flatnonzero(sims[row] >= sim_thr)
        for k in strong:
            t = int(train[int(k)])
            if phash_distance(records[c].phash, records[t].phash) <= ph_thr:
                bad = True; break
        if not bad and records[c].burst_id in train_by_burst:
            if max(float(embeddings[c] @ embeddings[t]) for t in train_by_burst[records[c].burst_id]) >= burst_thr:
                bad = True
        if not bad: keep.append(c)
    return keep, max_sim


def write_debug_csv(
    path: Path,
    all_records: Sequence[Record],
    compact_records: Sequence[Record],
    embeddings: np.ndarray,
    raw_categories: np.ndarray,
    category_specs: Sequence[SemanticSpec],
    raw_penalties: np.ndarray,
    penalty_specs: Sequence[SemanticSpec],
    penalty_scores: np.ndarray,
    eligible: Sequence[int],
    train: Sequence[int],
    validate: Sequence[int],
) -> None:
    """Write one compact per-image tuning table; does not affect selection."""
    path.parent.mkdir(parents=True, exist_ok=True)

    compact_pos = {r.path: i for i, r in enumerate(compact_records)}
    eligible_set, train_set, val_set = set(eligible), set(train), set(validate)

    selected = list(train) + list(validate)
    selected_arr = np.asarray(selected, np.int64)
    selected_set = set(selected)

    nearest_sim = np.full(len(compact_records), np.nan, np.float32)
    nearest_name = [""] * len(compact_records)
    if len(selected_arr):
        sims = embeddings @ embeddings[selected_arr].T
        for i in range(len(compact_records)):
            row = sims[i].copy()
            if i in selected_set:
                own = np.flatnonzero(selected_arr == i)
                if len(own):
                    row[int(own[0])] = -1.0
            j = int(np.argmax(row))
            if row[j] > -0.5:
                nearest_sim[i] = float(row[j])
                nearest_name[i] = Path(compact_records[int(selected_arr[j])].path).name

    fields = [
        "file", "path", "status", "reject_reason",
        "width", "height", "min_side", "aspect_ratio",
        "capture_source", "burst_id",
        "phash", "laplacian", "exposure",
        "blur_global", "blur_center", "blur_local",
        "prequality", "focus_risk", "brisque", "quality",
        "semantic_penalty",
        "nearest_selected_similarity", "nearest_selected_file",
    ]
    fields += [f"cat:{s.name}" for s in category_specs]
    fields += [f"pen:{s.name}" for s in penalty_specs]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in all_records:
            p = compact_pos.get(r.path)
            if p is None:
                status = "rejected"
            elif p in train_set:
                status = "train"
            elif p in val_set:
                status = "validate"
            elif p in eligible_set:
                status = "eligible_not_selected"
            else:
                status = "rejected"

            row = {
                "file": Path(r.path).name,
                "path": r.path,
                "status": status,
                "reject_reason": r.reject_reason,
                "width": r.width,
                "height": r.height,
                "min_side": r.min_side,
                "aspect_ratio": r.aspect_ratio,
                "capture_source": r.capture_source,
                "burst_id": r.burst_id,
                "phash": r.phash,
                "laplacian": r.laplacian,
                "exposure": r.exposure,
                "blur_global": r.blur_global,
                "blur_center": r.blur_center,
                "blur_local": r.blur_local,
                "prequality": r.prequality,
                "focus_risk": (r.focus_risk if p is not None else ""),
                "brisque": (r.brisque if p is not None else ""),
                "quality": (r.quality if p is not None else ""),
                "semantic_penalty": (float(penalty_scores[p]) if p is not None else ""),
                "nearest_selected_similarity": (float(nearest_sim[p]) if p is not None and np.isfinite(nearest_sim[p]) else ""),
                "nearest_selected_file": (nearest_name[p] if p is not None else ""),
            }

            if p is not None:
                for j, s in enumerate(category_specs):
                    row[f"cat:{s.name}"] = float(raw_categories[p, j])
                for j, s in enumerate(penalty_specs):
                    row[f"pen:{s.name}"] = float(raw_penalties[p, j])

            writer.writerow(row)


# ------------------------------ output ---------------------------------------

def materialize(records: Sequence[Record], ids: Sequence[int], folder: Path, mode: str) -> None:
    if folder.exists(): shutil.rmtree(folder)
    folder.mkdir(parents=True)
    used: set[str] = set()
    for i in ids:
        src = Path(records[i].path); name = src.name
        if name.lower() in used:
            suffix = hashlib.sha1(str(src).encode()).hexdigest()[:8]
            name = f"{src.stem}__{suffix}{src.suffix}"
        used.add(name.lower()); dst = folder/name
        if mode == "copy": shutil.copy2(src, dst)
        elif mode == "hardlink": os.link(src, dst)
        else: os.symlink(src, dst)


def print_coverage(label: str, selected: Sequence[int], raw: np.ndarray, specs: Sequence[SemanticSpec], targets: np.ndarray, cfg: Mapping[str, Any]) -> None:
    ev = evidence_transform(raw, cfg)
    print(f"\n{label} coverage (soft-equivalent selected / effective target):")
    for j, s in enumerate(specs):
        achieved = float(ev[np.asarray(selected, np.int64), j].sum()) if selected else 0.0
        if targets[j] >= .5 or achieved >= .5:
            print(f"  {s.name:24s} {achieved:6.1f} / {float(targets[j]):5.1f}")


# ------------------------------ main -----------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("identity_curator.json"))
    ns = ap.parse_args(argv)
    cfg = load_config(ns.config)
    source, output = ns.source.resolve(), ns.output.resolve()
    if not source.is_dir(): raise RuntimeError(f"Source is not a directory: {source}")
    if source == output: raise RuntimeError("Output must differ from source")
    device = resolve_device(str(cfg["runtime"]["device"]))
    print(f"Device: {device}", file=sys.stderr)

    paths = image_paths(source, output)
    if not paths: raise RuntimeError("No images found")
    print(f"Audit: {len(paths)} images", file=sys.stderr)
    workers = int(cfg["runtime"]["workers"])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(lambda p: audit_one(p, cfg), paths))
    assign_bursts(records, cfg)
    basic = [i for i, r in enumerate(records) if not r.reject_reason]
    if not basic: raise RuntimeError("No images survive basic audit")
    assign_prequality(records, basic, cfg)
    print(f"  basic eligible: {len(basic)}", file=sys.stderr)

    phash_ids = phash_dedup(records, basic, cfg)
    print(f"  pHash survivors: {len(phash_ids)}", file=sys.stderr)
    emb0 = dino_embeddings([records[i].path for i in phash_ids], cfg["models"]["embedding"], device, int(cfg["runtime"]["batch_size"]))
    semantic_ids, emb1 = semantic_dedup(records, phash_ids, emb0, cfg)
    print(f"  DINO near-duplicate survivors: {len(semantic_ids)}", file=sys.stderr)

    # From here on, position == row in emb/WD14 arrays. Keep compact arrays and a compact record list.
    compact_records = [records[i] for i in semantic_ids]
    embeddings = emb1
    categories, penalties = semantic_specs(cfg)
    all_specs = categories + penalties
    vocab = load_wd14_vocab(cfg)
    resolved = resolve_semantics(vocab, all_specs)
    print(f"WD14 vocabulary contract: {len(categories)} categories + {len(penalties)} penalties verified", file=sys.stderr)
    raw_all = wd14_scores(compact_records, list(range(len(compact_records))), cfg, all_specs, vocab, resolved, device)
    raw_cat = raw_all[:, :len(categories)]
    raw_pen = raw_all[:, len(categories):]
    pindex = {s.name: j for j, s in enumerate(penalties)}
    focus_risk, focus_hard = focus_assessment(compact_records, list(range(len(compact_records))), raw_pen[:, pindex["semantic_blur"]], cfg)
    semantic_hard = hard_semantic_reject(raw_pen, penalties)
    for p, r in enumerate(compact_records):
        if focus_hard[p]:
            r.reject_reason = "focus"
        elif semantic_hard[p]:
            r.reject_reason = "semantic_hard"
    eligible = np.flatnonzero(~focus_hard & ~semantic_hard).tolist()
    if len(eligible) < int(cfg["dataset"]["train"]):
        raise RuntimeError(f"Only {len(eligible)} technically eligible images remain; need {cfg['dataset']['train']}")
    print(f"Eligibility: {len(eligible)} pass; {int(focus_hard.sum())} focus rejects; {int((semantic_hard & ~focus_hard).sum())} semantic hard rejects", file=sys.stderr)

    brisque_ok = compute_brisque(compact_records, eligible, device, bool(cfg["audit"]["brisque"]["enabled"]))
    assign_quality(compact_records, eligible, cfg, brisque_ok)
    penalties_score = semantic_penalties(raw_pen, penalties, cfg)
    # Ineligible rows are never selected; quality value does not matter there.

    train, train_cov, train_targets = select_greedy(compact_records, eligible, embeddings, raw_cat, categories, penalties_score, int(cfg["dataset"]["train"]), cfg)
    if len(train) < int(cfg["dataset"]["train"]): raise RuntimeError("Could not fill training target")
    train_bursts: dict[str, int] = {}
    for i in train:
        if compact_records[i].burst_id: train_bursts[compact_records[i].burst_id] = train_bursts.get(compact_records[i].burst_id, 0)+1
    vpool, initial_sim = validation_pool(compact_records, eligible, train, embeddings, cfg)
    val_target = min(int(cfg["dataset"]["validate"]), len(vpool))
    validate, val_cov, val_targets = select_greedy(compact_records, vpool, embeddings, raw_cat, categories, penalties_score, val_target, cfg, initial_similarity=initial_sim, initial_burst_counts=train_bursts)
    if val_target < int(cfg["dataset"]["validate"]):
        print(f"Warning: only {val_target} validation images available after leakage filtering", file=sys.stderr)

    output.mkdir(parents=True, exist_ok=True)
    materialize(compact_records, train, output/"train", str(cfg["runtime"]["output_mode"]))
    materialize(compact_records, validate, output/"validate", str(cfg["runtime"]["output_mode"]))
    write_debug_csv(
        output/"debug.csv",
        records, compact_records, embeddings,
        raw_cat, categories, raw_pen, penalties, penalties_score,
        eligible, train, validate,
    )

    print(f"\nSelected train={len(train)}, validate={len(validate)}")
    print(f"Focus risk selected: train max={max(compact_records[i].focus_risk for i in train):.3f}, validate max={(max((compact_records[i].focus_risk for i in validate), default=0.0)):.3f}")
    print_coverage("Train", train, raw_cat, categories, train_targets, cfg)
    if validate: print_coverage("Validate", validate, raw_cat, categories, val_targets, cfg)
    print(f"\nOutput: {output/'train'} and {output/'validate'}")
    print(f"Debug: {output/'debug.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
