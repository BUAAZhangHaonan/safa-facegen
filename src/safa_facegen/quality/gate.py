"""Conservative engineering gates, NOT a universal face-quality standard.
Normalized input must be assembled from actual completed review evidence.
Missing fields mean REVIEW_REQUIRED, never zero defects. No weighted aggregate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import math


def select_better(candidate: dict, baseline: dict) -> dict:
    """Select a practically improved candidate, without granting release approval.

    ``manual_labels`` must contain the actual signed 256-prefix rows in each
    normalized input. ``manual`` counts must agree with these rows. A signed
    ``texture_comparison`` on the candidate refers to the exact baseline and is
    independent of TOPIQ scores. When fixed TOPIQ anchors are unavailable, a
    signed original-resolution texture review remains valid evidence.
    """
    missing, reasons, criteria = [], [], {}

    def result(decision):
        return {"decision": decision, "quality_approved": False,
                "missing": missing, "reasons": reasons, "criteria": criteria}

    if not isinstance(candidate, dict) or not isinstance(baseline, dict):
        missing.append("candidate_and_baseline_evidence")
        return result("REVIEW_REQUIRED")
    for key in ("model_id", "protocol_id"):
        if not candidate.get(key) or candidate.get(key) != baseline.get(key):
            missing.append("matching_" + key)
    for role, value in (("candidate", candidate), ("baseline", baseline)):
        if not isinstance(value.get("checkpoint_id"), str) or not value["checkpoint_id"].strip():
            missing.append(role + ".checkpoint_id")
        for key in ("samples", "nonfinite", "blank", "zero_face", "filtered"):
            item = value.get(key)
            if type(item) is not int or item < 0:
                missing.append(role + "." + key)
        for key in ("kid", "coverage"):
            item = value.get(key)
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                missing.append(role + "." + key)
        if isinstance(value.get("coverage"), (int, float)) and not 0 <= value["coverage"] <= 1:
            missing.append(role + ".coverage_range")
        if value.get("texture_review") not in ("acceptable", "rejected") or not str(value.get("texture_reviewer") or "").strip():
            missing.append(role + ".signed_texture_review")
    normalized = {}
    for role, value in (("candidate", candidate), ("baseline", baseline)):
        rows = value.get("manual_labels")
        if not isinstance(rows, list) or len(rows) != 256 or any(not isinstance(r, dict) for r in rows):
            missing.append(role + ".256_signed_manual_labels")
            continue
        indices = [r.get("sample_index") for r in rows]
        if any(type(i) is not int for i in indices) or sorted(indices) != list(range(256)):
            missing.append(role + ".manual_indices")
            continue
        if any(r.get("label") not in ("acceptable", "minor", "severe", "uncertain")
               or not isinstance(r.get("reviewer"), str) or not r["reviewer"].strip()
               or type(r.get("texture_issue")) is not bool or type(r.get("background_multi")) is not bool
               for r in rows):
            missing.append(role + ".complete_signed_manual_fields")
            continue
        counts = {k: sum(r["label"] == k for r in rows) for k in ("acceptable", "minor", "severe", "uncertain")}
        expected = {"reviewed": 256, **counts}
        actual = value.get("manual")
        if not isinstance(actual, dict) or any(type(actual.get(k)) is not int or actual[k] != v for k, v in expected.items()):
            missing.append(role + ".manual_counts_match_labels")
            continue
        if counts["uncertain"]:
            missing.append(role + ".resolve_uncertain_labels")
        normalized[role] = {"counts": counts, "severe_indices": {r["sample_index"] for r in rows if r["label"] == "severe"}}
    comparison = candidate.get("texture_comparison")
    if (not isinstance(comparison, dict) or comparison.get("verdict") not in ("improved", "unchanged", "worse")
            or not isinstance(comparison.get("reviewer"), str) or not comparison["reviewer"].strip()
            or comparison.get("baseline_checkpoint_id") != baseline.get("checkpoint_id")):
        missing.append("signed_texture_comparison_to_exact_baseline")
    if missing:
        return result("REVIEW_REQUIRED")
    if candidate["samples"] != 1024 or baseline["samples"] != 1024:
        reasons.append("wrong_sample_count")
    for key in ("nonfinite", "blank", "zero_face", "filtered"):
        if candidate[key] != 0:
            reasons.append("candidate_" + key)
    criteria["kid_limit"] = baseline["kid"] + max(.0005, .10 * abs(baseline["kid"]))
    criteria["coverage_floor"] = baseline["coverage"] - .05
    if candidate["kid"] > criteria["kid_limit"]:
        reasons.append("kid_distribution_regression")
    if candidate["coverage"] < criteria["coverage_floor"]:
        reasons.append("coverage_regression_over_5pp")
    if comparison["verdict"] == "worse" or (baseline["texture_review"] == "acceptable" and candidate["texture_review"] == "rejected"):
        reasons.append("texture_regression")
    old, new = normalized["baseline"]["counts"], normalized["candidate"]["counts"]
    if new["severe"] > old["severe"]:
        reasons.append("serious_subject_defect_count_increased")
    criteria.update(severe_before=old["severe"], severe_after=new["severe"],
                    minor_before=old["minor"], minor_after=new["minor"])
    if old["severe"] >= 4:
        required = max(2, math.ceil(.25 * old["severe"]))
        criteria["required_severe_reduction"] = required
        improved = old["severe"] - new["severe"] >= required
    elif old["severe"] > 0:
        criteria["required_severe_reduction"] = 1
        improved = old["severe"] - new["severe"] >= 1
    else:
        criteria["required_minor_reduction_if_no_texture_improvement"] = 3
        improved = new["severe"] == 0 and old["minor"] - new["minor"] >= 3
        if not improved and new["severe"] == 0 and comparison["verdict"] == "improved":
            if comparison.get("basis") == "original_resolution_review":
                improved = True
            elif comparison.get("calibration_valid") is not True or not isinstance(comparison.get("calibration_id"), str) or not comparison["calibration_id"].strip():
                missing.append("texture_improvement_review_basis")
            else:
                improved = True
    if reasons:
        return result("NOT_BETTER")
    if missing:
        return result("REVIEW_REQUIRED")
    if not improved:
        reasons.append("no_practical_manual_quality_improvement")
        return result("NOT_BETTER")
    return result("BETTER")


def assess(candidate: dict, baseline: dict | None = None) -> dict:
    missing, reasons = [], []
    for field in ("samples", "nonfinite", "blank", "zero_face", "filtered", "manual", "kid", "coverage"):
        if field not in candidate: missing.append(field)
    if missing:
        return {"decision":"REVIEW_REQUIRED","missing":missing,"quality_approved":False}
    for key in ("samples","nonfinite","blank","zero_face","filtered"):
        value=candidate[key]
        if type(value) is not int or value<0:
            return {"decision":"REVIEW_REQUIRED","invalid":[key],"quality_approved":False}
    if candidate["samples"]!=1024: reasons.append("wrong_sample_count")
    for key in ("nonfinite","blank","zero_face","filtered"):
        if candidate[key]!=0: reasons.append(key)
    m=candidate["manual"]
    if not isinstance(m,dict) or any(type(m.get(k)) is not int or m[k]<0 for k in
          ("reviewed","acceptable","minor","severe","uncertain")):
        return {"decision":"REVIEW_REQUIRED","missing":["complete_manual_counts"],"quality_approved":False}
    if m["reviewed"]!=256 or sum(m[k] for k in ("acceptable","minor","severe","uncertain"))!=256:
        return {"decision":"REVIEW_REQUIRED","missing":["256_signed_labels"],"quality_approved":False}
    if m["uncertain"]:
        return {"decision":"REVIEW_REQUIRED","missing":["resolve_uncertain_labels"],"quality_approved":False}
    rows = candidate.get("manual_labels")
    if not isinstance(rows, list) or len(rows) != 256 or any(not isinstance(r, dict) for r in rows):
        return {"decision":"REVIEW_REQUIRED","missing":["256_actual_signed_labels"],"quality_approved":False}
    indices = [r.get("sample_index") for r in rows]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(256)):
        return {"decision":"REVIEW_REQUIRED","missing":["manual_indices"],"quality_approved":False}
    if any(r.get("label") not in ("acceptable", "minor", "severe", "uncertain")
           or not isinstance(r.get("reviewer"), str) or not r["reviewer"].strip()
           or type(r.get("texture_issue")) is not bool or type(r.get("background_multi")) is not bool
           for r in rows):
        return {"decision":"REVIEW_REQUIRED","missing":["complete_signed_manual_fields"],"quality_approved":False}
    if any(sum(r["label"] == label for r in rows) != m[label]
           for label in ("acceptable", "minor", "severe", "uncertain")):
        return {"decision":"REVIEW_REQUIRED","missing":["manual_counts_match_labels"],"quality_approved":False}
    if m["severe"]>0: reasons.append("serious_subject_defects")
    if m["minor"]>12: reasons.append("minor_defects_over_12_of_256")
    for key in ("kid","coverage"):
        value=candidate[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            return {"decision":"REVIEW_REQUIRED","invalid":[key],"quality_approved":False}
    if not 0 <= candidate["coverage"] <= 1:
        return {"decision":"REVIEW_REQUIRED","invalid":["coverage_range"],"quality_approved":False}
    if baseline:
        if not candidate.get("model_id") or baseline.get("model_id") != candidate["model_id"]:
            return {"decision":"REVIEW_REQUIRED","missing":["same_model_baseline"],"quality_approved":False}
        if baseline.get("protocol_id")!=candidate.get("protocol_id") or not candidate.get("protocol_id"):
            return {"decision":"REVIEW_REQUIRED","missing":["same_distribution_protocol"],"quality_approved":False}
        if any(isinstance(baseline.get(k),bool) or not isinstance(baseline.get(k),(float,int)) or not math.isfinite(baseline[k]) for k in ("kid","coverage")):
            return {"decision":"REVIEW_REQUIRED","missing":["baseline_distribution_fields"],"quality_approved":False}
        if not 0 <= baseline["coverage"] <= 1:
            return {"decision":"REVIEW_REQUIRED","invalid":["baseline_coverage_range"],"quality_approved":False}
        if candidate["kid"]>baseline["kid"]+max(.0005,.10*abs(baseline["kid"])):
            reasons.append("kid_distribution_regression")
        if candidate["coverage"]<baseline["coverage"]-.05:
            reasons.append("coverage_regression_over_5pp")
    else:
        missing.append("same_model_baseline_distribution")
    texture=candidate.get("texture_review")
    if not isinstance(candidate.get("texture_reviewer"), str) or not candidate["texture_reviewer"].strip():
        missing.append("signed_texture_review")
    if texture not in ("acceptable", "rejected"):
        missing.append("texture_review")
    elif texture=="rejected": reasons.append("texture_or_blur_rejected")
    if reasons:
        return {"decision":"NOT_RELEASE_READY","reasons":reasons,"missing":missing,"quality_approved":False}
    if missing:
        return {"decision":"REVIEW_REQUIRED","missing":missing,"quality_approved":False}
    return {"decision":"READY_FOR_OWNER_ACCEPTANCE","quality_approved":False,
            "statement":"Fixed 256-prefix gate passed; no population-wide zero-defect promise"}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("candidate",type=Path);p.add_argument("--baseline",type=Path)
    p.add_argument("--select-better", action="store_true", help="Compare practical improvement instead of release readiness")
    a=p.parse_args()
    if a.select_better and not a.baseline:
        p.error("--select-better requires --baseline")
    gate = select_better if a.select_better else assess
    print(json.dumps(gate(json.loads(a.candidate.read_text()),
        json.loads(a.baseline.read_text()) if a.baseline else None),ensure_ascii=False,indent=2))
