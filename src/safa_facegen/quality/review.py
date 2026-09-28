"""Offline quality tools for EXISTING review/images/sample-*.png.
prepare: writes a full-resolution review page and EMPTY labels, never inferred labels.
score: computes TOPIQ-NR only; no face repair, filtering or extra generation.
summarize: counts complete manual labels; missing/uncertain blocks release readiness.
"""
from __future__ import annotations
import argparse
import csv
import html
import importlib.metadata
import json
from pathlib import Path
import os
import numpy as np

LABELS = ("acceptable", "minor", "severe", "uncertain")
FIELDS = ("sample_index", "path", "label", "texture_issue", "background_multi", "reviewer", "note")


def reviewed_paths(review: Path):
    review = review.resolve()
    metadata = json.loads((review / "summary.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("Use a completed review; do not score partial transfers")
    generation = metadata.get("generation", {})
    if generation.get("nonfinite") != 0 or generation.get("filtered_samples") != 0:
        raise ValueError("Nonfinite/filtered review cannot receive a replacement quality score")
    files = sorted((review / "images").glob("sample-*.png"))
    if len(files) != 1024 or [p.name for p in files] != [f"sample-{i:06d}.png" for i in range(1024)]:
        raise ValueError("Expected all 1024 original images in original order")
    return files, metadata


def write_new(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        out.write(text)


def prepare(review: Path, output: Path):
    paths, meta = reviewed_paths(review)
    output.mkdir(parents=True, exist_ok=True)
    rows = [{k: "" for k in FIELDS} for _ in range(256)]
    cards = []
    for index, path in enumerate(paths[:256]):
        relative = Path(os.path.relpath(path, output)).as_posix()
        rows[index].update(sample_index=index, path=str(path))
        cards.append(f'<section data-i="{index}"><h3>#{index:03d}</h3>'
                     f'<a href="{html.escape(relative, quote=True)}" target="_blank">'
                     f'<img src="{html.escape(relative, quote=True)}" width="256" height="256"></a>'
                     '<p><select class="label"><option value="">未审阅</option>'
                     '<option value="acceptable">可接受</option><option value="minor">轻微缺陷</option>'
                     '<option value="severe">严重主体缺陷</option><option value="uncertain">不确定</option></select></p>'
                     '<p><label><input class="texture" type="checkbox">纹理/模糊问题</label>'
                     '<label><input class="background" type="checkbox">背景多人脸</label></p>'
                     '<input class="note" placeholder="具体部位/原因；自然牙列、表情、遮挡不自动判错"></section>')
    with (output / "labels-empty.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    metadata_text = html.escape(json.dumps({k: meta.get(k) for k in
                      ("model_id", "checkpoint", "started_at_utc")}, ensure_ascii=False))
    safe_rows = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAFA 256 图质量审阅</title>
<style>body{font-family:sans-serif;margin:24px}header{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #aaa}.grid{display:grid;grid-template-columns:repeat(auto-fit,280px);gap:16px}section{border:1px solid #aaa;padding:10px}input.note{width:250px}p{font-size:13px}img{object-fit:contain}</style>
<header><b>固定前 256 张；点图看原尺寸；不挑图</b><p>MODEL_META</p>
审阅人 <input id="who"><button onclick="download()">导出 labels-reviewed.csv</button>
<p>主体严重缺陷、轻微细节、背景多脸分别记录。没有审阅的项保持空白。</p></header><main class="grid">CARDS</main>
<script>const rows=ROWS;
function csv(v){return '"'+String(v??'').replaceAll('"','""')+'"';}
function download(){document.querySelectorAll('section').forEach(s=>{let r=rows[Number(s.dataset.i)]; r.label=s.querySelector('.label').value;r.texture_issue=s.querySelector('.texture').checked?'true':'false';r.background_multi=s.querySelector('.background').checked?'true':'false';r.reviewer=document.getElementById('who').value.trim();r.note=s.querySelector('.note').value;});
let keys=Object.keys(rows[0]);let text='\\ufeff'+keys.join(',')+'\\n'+rows.map(r=>keys.map(k=>csv(r[k])).join(',')).join('\\n');let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));a.download='labels-reviewed.csv';a.click();URL.revokeObjectURL(a.href);}</script></html>'''
    page = page.replace("MODEL_META", metadata_text).replace("CARDS", "\n".join(cards)).replace("ROWS", safe_rows)
    write_new(output / "review-256.html", page)


def load_topiq(weights: Path, device: str):
    """Load a complete official TOPIQ checkpoint offline, including its backbone."""
    import torch
    import pyiqa
    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(weights)
    metric = pyiqa.create_metric("topiq_nr", device=device, pretrained=False,
                                backbone_pretrain=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    metric.net.load_state_dict(state["params"], strict=True)
    metric.eval()
    if bool(metric.lower_better):
        raise RuntimeError("Unexpected TOPIQ-NR score direction")
    return metric


def score(review: Path, output: Path, device: str = "cpu", batch: int = 16,
          weights: Path | None = None, metric=None):
    import torch
    from PIL import Image
    if batch < 1:
        raise ValueError("Batch must be positive")
    paths, meta = reviewed_paths(review)
    output.mkdir(parents=True, exist_ok=True)
    if weights is None:
        raise ValueError("An explicit local official TOPIQ-NR checkpoint is required")
    metric = metric if metric is not None else load_topiq(weights, device)
    if bool(metric.lower_better):
        raise RuntimeError("Unexpected TOPIQ-NR score direction; inspect installed model")
    values = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch):
            tensors = []
            for path in paths[start:start+batch]:
                with Image.open(path) as image:
                    if image.size != (256, 256):
                        raise ValueError("Native review resolution must remain 256x256")
                    arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.
                tensors.append(torch.from_numpy(arr).permute(2,0,1))
            x = torch.stack(tensors).to(device)
            y = metric(x).reshape(-1).detach().cpu().numpy()
            if len(y) != len(tensors) or not np.isfinite(y).all():
                raise RuntimeError("IQA returned invalid results; do not impute missing values")
            values.extend(float(v) for v in y)
    with (output / "topiq-scores.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["sample_index", "path", "topiq_nr"])
        writer.writerows((i, str(path), value) for i,(path,value) in enumerate(zip(paths, values)))
    result = {"model_id":meta.get("model_id"), "checkpoint":meta.get("checkpoint"),
              "metric":"topiq_nr", "higher_is_better":True, "image_count":1024,
              "input":"original RGB 256x256 float [0,1]; no external resize or repair",
              "pyiqa_version":importlib.metadata.version("pyiqa"),
              "torch_version":torch.__version__, "device":device, "batch_size":batch,
              "weights_path":str(weights.resolve()), "weights_bytes":weights.stat().st_size,
              "weight_loading":"complete state including backbone; strict=True; offline",
              "mean":float(np.mean(values)), "median":float(np.median(values)),
              "p10":float(np.quantile(values,.1)), "p05":float(np.quantile(values,.05)),
              "anatomical_validity":"not_measured_by_TOPIQ", "calibration":"required_for_thresholds"}
    write_new(output / "topiq-summary.json", json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result


def summarize_labels(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 256 or sorted(int(r["sample_index"]) for r in rows) != list(range(256)):
        raise ValueError("Labels must cover the unchanged first 256 once each")
    unknown = [r for r in rows if r.get("label", "") not in LABELS or not r.get("reviewer", "").strip()
               or r.get("texture_issue", "").lower() not in ("true", "false")
               or r.get("background_multi", "").lower() not in ("true", "false")]
    counts = {name:sum(r.get("label")==name for r in rows) for name in LABELS}
    return {"expected":256,"reviewed":256-len(unknown),"missing_or_unsigned":len(unknown),
            **counts, "background_multi":sum(r.get("background_multi", "").lower()=="true" for r in rows),
            "texture_issue":sum(r.get("texture_issue", "").lower()=="true" for r in rows),
            "complete":not unknown and counts["uncertain"]==0,
            "rates_are_fixed_prefix_counts_not_population_guarantees":True}


def calibrate(score_csv: Path, labels_csv: Path):
    """One-time local anchor calibration. No anatomical classification claim.
    Need >=32 explicitly acceptable texture-clean anchors and >=16 texture-failed
    anchors. AUC<.75 => auxiliary-only. Threshold is clean-anchor 5th percentile.
    Apply one fixed calibration to subsequent candidate reviews; do not refit it
    per checkpoint. This is engineering QC, not an IQA benchmark study.
    """
    with score_csv.open(encoding="utf-8-sig", newline="") as f:
        scores = {r["sample_index"]:float(r["topiq_nr"]) for r in csv.DictReader(f)}
    with labels_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    good, bad = [], []
    for r in rows:
        if not r.get("reviewer", "").strip() or r.get("label") not in LABELS:
            continue
        value = scores.get(r["sample_index"])
        if value is None or not np.isfinite(value):
            continue
        if r.get("label")=="acceptable" and r.get("texture_issue", "").lower()=="false":
            good.append(value)
        elif r.get("texture_issue", "").lower()=="true":
            bad.append(value)
    if len(good)<32 or len(bad)<16:
        return {"valid_for_gate":False,"reason":"insufficient human texture anchors", "good":len(good),"bad":len(bad)}
    diffs = np.asarray(good)[:,None]-np.asarray(bad)[None,:]
    auc = float(np.mean(diffs>0)+0.5*np.mean(diffs==0))
    return {"valid_for_gate":auc>=.75,"auc_on_local_anchors":auc,"threshold":float(np.quantile(good,.05)),
            "good":len(good),"bad":len(bad),"rule":"lower than fixed acceptable-anchor p05 is flagged",
            "scope":"texture and clarity only; never substitute for structure review"}


def apply_calibration(score_csv: Path, calibration_path: Path):
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("valid_for_gate") is not True:
        return {"status":"AUXILIARY_ONLY", "low_score_rate":None,
                "reason":"No valid fixed human-anchor calibration"}
    threshold = calibration.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not np.isfinite(threshold):
        raise ValueError("Invalid calibration threshold")
    with score_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1024 or sorted(int(r["sample_index"]) for r in rows) != list(range(1024)):
        raise ValueError("Scores must cover all 1024 retained images exactly once")
    values = np.asarray([float(r["topiq_nr"]) for r in rows])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite TOPIQ scores")
    return {"status":"complete", "calibration_source":str(calibration_path.resolve()),
            "threshold":threshold, "low_score_count":int((values < threshold).sum()),
            "low_score_rate":float((values < threshold).mean()),
            "scope":"texture and clarity only; structural review still required"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for name in ("prepare","score"):
        sub=commands.add_parser(name);sub.add_argument("--review",required=True,type=Path);sub.add_argument("--out",required=True,type=Path)
        if name=="score":
            sub.add_argument("--device",default="cpu");sub.add_argument("--batch",type=int,default=16)
            sub.add_argument("--weights",required=True,type=Path)
    sub=commands.add_parser("summarize");sub.add_argument("--labels",required=True,type=Path)
    sub=commands.add_parser("calibrate");sub.add_argument("--labels",required=True,type=Path);sub.add_argument("--scores",required=True,type=Path)
    sub=commands.add_parser("apply-calibration");sub.add_argument("--scores",required=True,type=Path);sub.add_argument("--calibration",required=True,type=Path)
    args=parser.parse_args()
    if args.command=="prepare": prepare(args.review,args.out)
    elif args.command=="score": print(json.dumps(score(args.review,args.out,args.device,args.batch,args.weights),ensure_ascii=False,indent=2))
    elif args.command=="summarize": print(json.dumps(summarize_labels(args.labels),ensure_ascii=False,indent=2))
    elif args.command=="calibrate": print(json.dumps(calibrate(args.scores,args.labels),ensure_ascii=False,indent=2))
    else: print(json.dumps(apply_calibration(args.scores,args.calibration),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
