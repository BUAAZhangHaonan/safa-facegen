"""Assemble quality evidence from existing reviews, without generating images."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .coverage import compute_prdc
from .gate import assess, select_better
from .review import reviewed_paths, summarize_labels, write_new


def assemble(review: Path, *, protocol_id: str, labels: Path | None = None,
             texture_review: str | None = None, texture_reviewer: str | None = None,
             texture_comparison: Path | None = None) -> dict:
    paths, summary = reviewed_paths(review)
    record_path = review / "training-record.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    if record and record.get("model_id") != summary["model_id"]:
        raise ValueError("Training record and review model identities differ")
    identity_sources = [summary, summary.get("ema_identity") or {}, record]
    identity = {}
    for key in ("checkpoint_id", "objective_id", "stage_id", "stage_complete"):
        known = [source[key] for source in identity_sources if source.get(key) is not None]
        if known and any(value != known[0] for value in known):
            raise ValueError(f"Conflicting review/training identity: {key}")
        identity[key] = known[0] if known else None
    distribution = summary.get("distribution", {})
    if distribution.get("status") != "complete":
        raise ValueError("Distribution evaluation is not complete")
    if not protocol_id.strip():
        raise ValueError("Register the exact distribution/sampling protocol before comparison")
    features = []
    for kind in ("real", "generated"):
        path = review / f"{kind}-inception.npy"
        if not path.exists():
            path = review / f"{kind}-inception-features.npy"
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        if value.shape != (1024, 2048):
            raise ValueError(f"Expected original 1024 x 2048 individual features: {path}")
        features.append(value)
    prdc = compute_prdc(*features)
    manual = None
    manual_labels = None
    if labels:
        with labels.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            index = int(row["sample_index"])
            if not 0 <= index < 256 or Path(row["path"]).resolve() != paths[index]:
                raise ValueError("Manual label image identity differs from this review")
        manual = summarize_labels(labels)
        def optional_bool(value):
            return {"true": True, "false": False}.get((value or "").strip().lower())
        manual_labels = [{"sample_index": int(row["sample_index"]), "path": row["path"],
                          "label": row.get("label"), "reviewer": row.get("reviewer"),
                          "texture_issue": optional_bool(row.get("texture_issue")),
                          "background_multi": optional_bool(row.get("background_multi")),
                          "note": row.get("note")} for row in rows]
    if texture_review is not None and not (texture_reviewer or "").strip():
        raise ValueError("Texture judgment requires an explicit reviewer")
    comparison = None
    if texture_comparison is not None:
        comparison = json.loads(texture_comparison.read_text(encoding="utf-8"))
        if not isinstance(comparison, dict):
            raise ValueError("Texture comparison must be an explicit human-record JSON object")
    generation = summary.get("generation", {})
    faces = summary.get("face_detection", {})
    if generation.get("status") != "complete" or faces.get("status") != "complete":
        raise ValueError("Generation or face detection evidence is incomplete")
    return {
        "model_id": summary["model_id"], "checkpoint": summary["checkpoint"], **identity,
        "review_directory": str(review.resolve()), "protocol_id": protocol_id,
        "samples": generation.get("samples"), "nonfinite": generation.get("nonfinite"),
        "blank": generation.get("blank_spatial_std_lt1"), "zero_face": faces.get("zero"),
        "filtered": generation.get("filtered_samples"), "manual": manual, "manual_labels": manual_labels,
        "kid": distribution.get("kernel_inception_distance_mean"), "coverage": prdc["coverage"],
        "prdc": prdc, "texture_review": texture_review, "texture_reviewer": texture_reviewer,
        "texture_comparison": comparison,
        "texture_comparison_source": str(texture_comparison.resolve()) if texture_comparison else None,
        "manual_labels_source": str(labels.resolve()) if labels else None,
        "quality_approved": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--texture-review", choices=("acceptable", "rejected"))
    parser.add_argument("--texture-reviewer")
    parser.add_argument("--texture-comparison", type=Path, help="Explicit signed human texture comparison JSON; never generated by IQA")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    evidence = assemble(args.review, protocol_id=args.protocol_id, labels=args.labels,
                        texture_review=args.texture_review, texture_reviewer=args.texture_reviewer,
                        texture_comparison=args.texture_comparison)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    evidence["gate"] = assess(evidence, baseline)
    if baseline is not None:
        evidence["selection"] = select_better(evidence, baseline)
    write_new(args.out, json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
