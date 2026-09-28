"""Serial post-evaluation quality panel for registered quality-v1 objectives."""
from __future__ import annotations

import json
from pathlib import Path

from ..common import utc_now
from ..data import atomic_json

OBJECTIVES = {"imf_boundary_v1", "min_snr_epsilon_v1", "rf_batch_ot_v1",
              "lcf_real_v1", "lcd_teacher_v2"}
WEIGHTS = "models/evaluation/cfanet_nr_koniq_res50-9a73138b.pth"


def eligible(summary: dict) -> bool:
    identity = summary.get("ema_identity") or {}
    return (summary.get("status") == "complete" and summary.get("integrity_mode") == "metadata"
            and identity.get("objective_id") in OBJECTIVES and bool(identity.get("stage_id")))


def protocol(review: Path, summary: dict) -> dict:
    """Record concrete inputs without hashing, or pretending old protocols match."""
    reference = json.loads((review / "real-records.json").read_text(encoding="utf-8"))
    distribution = summary.get("distribution", {})
    required = ("sampling", "generation_precision", "noise_protocol", "seed")
    missing = [key for key in required if summary.get(key) is None]
    if missing:
        raise ValueError("Evaluation has no actual protocol fields: " + ", ".join(missing))
    indices = reference.get("indices")
    records = reference.get("records")
    if not isinstance(indices, list) or len(indices) != 1024 or not isinstance(records, list) or len(records) != 1024:
        raise ValueError("Expected the actual fixed 1024 reference records and indices")
    identities = []
    for record in records:
        if "id" not in record or "bytes" not in record:
            raise ValueError("Reference record identity is incomplete")
        identities.append({"id": record["id"], "bytes": record["bytes"]})
    return {"schema_version": 1, "model_id": summary["model_id"],
            "sampling": summary["sampling"], "generation_precision": summary["generation_precision"],
            "noise_protocol": summary["noise_protocol"], "seed": summary["seed"],
            "reference_seed": reference.get("seed"), "reference_indices": indices,
            "reference_records": identities,
            "feature_extractor": distribution.get("feature_extractor"),
            "feature_dimension": distribution.get("feature_dimension"),
            "torch_fidelity_version": distribution.get("torch_fidelity_version"),
            "registered_extractor_weights_identity": distribution.get("weights_sha256"),
            "kid_subsets": distribution.get("kid_subsets"),
            "kid_subset_size": distribution.get("kid_subset_size"),
            "kid_rng_seed": distribution.get("rng_seed")}


def run_panel(root: Path, review: Path, summary: dict, *, device: str = "cuda:0") -> dict | None:
    """Do not propagate panel failures into the already complete formal review.

    The caller runs this synchronously in its existing evaluation thread. No
    generation/transfer work is scheduled, and no manual decision is inferred.
    """
    if not eligible(summary):
        return None
    output = review / "quality-v1"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
    status = {"status": "running", "started_at_utc": utc_now(),
              "checkpoint_id": summary["ema_identity"].get("checkpoint_id"),
              "objective_id": summary["ema_identity"]["objective_id"],
              "formal_review_status": "complete", "quality_approved": False,
              "selection": "REVIEW_REQUIRED", "steps": {}, "errors": []}
    atomic_json(status_path, status)

    def step(name, operation):
        try:
            operation()
            status["steps"][name] = "complete"
        except Exception as exc:
            status["steps"][name] = "failed"
            status["errors"].append({"step": name, "error": f"{type(exc).__name__}: {exc}"})
        atomic_json(status_path, status)

    def human_page():
        from .review import prepare
        if not (output / "labels-empty.csv").exists() and not (output / "review-256.html").exists():
            prepare(review, output)
        elif not all((output / name).exists() for name in ("labels-empty.csv", "review-256.html")):
            raise ValueError("Partial human-review artifacts retained; do not overwrite labels")

    def topiq():
        from .review import score
        if not (output / "topiq-summary.json").exists():
            score(review, output, device, 16, root / WEIGHTS)

    def evidence():
        from .evidence import assemble
        from .gate import assess
        contract = protocol(review, summary)
        atomic_json(output / "protocol.json", contract)
        # Canonical protocol content is the identity; no digest or guessed match
        # to historical manually named protocols is introduced.
        protocol_id = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result = assemble(review, protocol_id=protocol_id)
        result["gate"] = assess(result)
        result["baseline_comparison"] = "requires_same_protocol_signed_baseline"
        atomic_json(output / "candidate-unreviewed.json", result)

    step("human_review_page", human_page)
    step("topiq_nr", topiq)
    step("coverage_and_evidence", evidence)
    status.update(status="complete" if not status["errors"] else "failed",
                  finished_at_utc=utc_now(), manual_labels="not_measured",
                  selection="REVIEW_REQUIRED")
    atomic_json(status_path, status)
    return status
