"""Reproducible data roles and provenance shared by research experiments.

This module deliberately has no ML dependencies so data audits run offline.
"""
import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROLES = ("selection", "tuning", "optimization", "evaluation")


def text_id(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Atomic, strict JSON. Nonfinite measurements must be handled explicitly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def records_from_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def canonical_records(records, require_labels=True):
    """Deduplicate normalized content; conflicting labels/IDs are errors."""
    by_content, by_id = {}, {}
    for raw in records:
        text = raw.get("input", raw.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every example needs nonempty input/text")
        content_id = text_id(text)
        record = dict(raw, input=text, content_id=content_id,
                      id=str(raw.get("id", content_id)))
        if require_labels and record.get("label") not in (0, 1):
            raise ValueError("Membership labels must be 0 or 1")
        if record["id"] in by_id and by_id[record["id"]] != content_id:
            raise ValueError(f"ID reused for different content: {record['id']}")
        if content_id in by_content:
            if by_content[content_id].get("label") != record.get("label"):
                raise ValueError("Duplicate content has conflicting membership labels")
            continue
        by_content[content_id] = record
        by_id[record["id"]] = content_id
    return sorted(by_content.values(), key=lambda r: r["id"])


def assert_disjoint(groups):
    seen_ids, seen_content = {}, {}
    for role, rows in groups.items():
        for row in rows:
            for seen, value in ((seen_ids, row["id"]),
                                (seen_content, text_id(row["input"]))):
                if value in seen:
                    raise ValueError(f"Overlap/duplicate between {seen[value]} and {role}: {row['id']}")
                seen[value] = role


def make_splits(records, seed=42, fractions=(0.2, 0.2, 0.4, 0.2)):
    if len(fractions) != 4 or any(f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("Provide four positive split fractions summing to one")
    records = canonical_records(records)
    groups = {role: [] for role in ROLES}
    rng = random.Random(seed)
    for label in (0, 1):
        rows = [r for r in records if r["label"] == label]
        if len(rows) < len(ROLES):
            raise ValueError("At least four distinct examples per class are required")
        rng.shuffle(rows)
        # Guarantee each class is present in each role, distribute the remainder.
        remaining = len(rows) - len(ROLES)
        counts = [1 + int(remaining * f) for f in fractions]
        order = sorted(range(4), key=lambda i: (-(remaining * fractions[i] % 1), i))
        for i in order[:len(rows) - sum(counts)]:
            counts[i] += 1
        offset = 0
        for role, count in zip(ROLES, counts):
            groups[role].extend(rows[offset:offset + count])
            offset += count
    for rows in groups.values():
        rows.sort(key=lambda r: r["id"])
    assert_disjoint(groups)
    return {"schema_version": 1, "seed": seed, "fractions": list(fractions),
            "dataset_hash": digest(records), "roles": groups,
            "membership_basis": "dataset_labels_unverified_for_backbone"}


def validate_splits(bundle):
    if set(bundle["roles"]) != set(ROLES):
        raise ValueError(f"Expected data roles {ROLES}")
    assert_disjoint(bundle["roles"])
    for role, rows in bundle["roles"].items():
        if {r.get("label") for r in rows} != {0, 1}:
            raise ValueError(f"Both membership classes required in {role}")
        for row in rows:
            if row.get("content_id") != text_id(row["input"]):
                raise ValueError(f"Content hash mismatch: {row['id']}")
    records = sorted([r for rows in bundle["roles"].values() for r in rows], key=lambda r: r["id"])
    if digest(records) != bundle["dataset_hash"]:
        raise ValueError("Split data fingerprint mismatch")
    return bundle


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def provenance(config, model=None):
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    versions = {}
    for name in ("torch", "transformers", "datasets", "numpy", "peft", "lm-eval"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "config_hash": digest(config), "git_commit": commit,
              "git_dirty": dirty, "packages": versions}
    if model is not None:
        result["resolved_model_commit"] = getattr(model.config, "_commit_hash", None)
        result["model_config"] = model.config.to_dict()
        result["dtype"] = str(next(model.parameters()).dtype)
        result["device"] = str(next(model.parameters()).device)
    return result
