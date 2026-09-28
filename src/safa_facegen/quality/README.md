# Offline quality evidence

Run these commands in the K100 evaluation environment on a completed review. They
reuse its 1024 unfiltered original images and individual Inception features. They
never generate, repair, rank-filter, or replace images. Historical `generated-inception.npy`
and `real-inception.npy` already contain individual features, not mean/covariance.

```bash
python -m safa_facegen.quality.review prepare --review "$REVIEW" --out "$REVIEW/quality-v1"
python -m safa_facegen.quality.review score --review "$REVIEW" --out "$REVIEW/quality-v1" --device cuda:0 --batch 16 --weights models/evaluation/cfanet_nr_koniq_res50-9a73138b.pth
python -m safa_facegen.quality.evidence --review "$REVIEW" --out "$REVIEW/quality-v1/candidate.json" --protocol-id "$REGISTERED_PROTOCOL"
```

TOPIQ uses the author's complete KonIQ ResNet50 checkpoint, loaded strictly
including the backbone. All network downloads are outside scoring; neither a
random backbone nor random missing keys can silently receive a score. It requires
the existing K100 PyTorch, pyiqa, Pillow and NumPy; coverage additionally requires
SciPy. No H100 dependency or numerical setting is changed by these tools.

`prepare` writes a native-resolution review page and **empty** labels. Signed
labels must cover the same first 256 image paths exactly. The four structural
labels are acceptable/minor/severe/uncertain; texture and background-face fields
remain separate. `evidence --labels ... --texture-review acceptable|rejected
--texture-reviewer ... --baseline ...` assembles actual evidence. Missing or
uncertain labels block readiness. A TOPIQ score alone never approves texture or
anatomy. Use `review calibrate --scores ... --labels ...` once with signed texture
anchors; insufficient anchors leave calibration invalid. Do not refit a threshold
per checkpoint.

After human review, the complete comparison command is:

```bash
python -m safa_facegen.quality.evidence --review "$REVIEW" --out "$REVIEW/quality-v1/candidate-reviewed.json" --protocol-id "$REGISTERED_PROTOCOL" --labels "$REVIEW/quality-v1/labels-reviewed.csv" --texture-review acceptable --texture-reviewer "$REVIEWER" --texture-comparison "$SIGNED_TEXTURE_COMPARISON_JSON" --baseline "$BASELINE_REVIEWED_JSON"
python -m safa_facegen.quality.gate "$REVIEW/quality-v1/candidate-reviewed.json" --baseline "$BASELINE_REVIEWED_JSON" --select-better
```

`--texture-comparison` is a human-authored JSON object containing `verdict`
(`improved`, `unchanged`, or `worse`), `reviewer`, `baseline_checkpoint_id`, and
`basis: "original_resolution_review"`. The actual verdict must come from review,
not this example or automatic TOPIQ scoring. Valid fixed-anchor calibration can
also support the texture improvement route. The assembler retains each signed
label and converts explicit CSV true/false fields to booleans; empty fields stay
unknown. Checkpoint, objective and stage identities are read from the actual
summary/EMA identity/training record, with conflicting identities rejected and
missing identities left null. It never infers stage completion from a round
number or a completed image evaluation. Selection returns `BETTER`, `NOT_BETTER`
or `REVIEW_REQUIRED`; release readiness and owner approval remain separate.

Distribution comparison requires the same model and explicit registered protocol
(sampling, precision, fixed real/reference order and extractor). Reuse a protocol
identifier only when those conditions truly match. KID and coverage are separate
gates. The strongest possible result is `READY_FOR_OWNER_ACCEPTANCE`, always with
`quality_approved: false`; owner acceptance remains external.

Artifacts are written with exclusive creation where practical. Keep original
reviews and signed labels; use a new evidence output name for subsequent human
revisions. Quality artifacts belong in ignored `reports`, not training history.

The existing replication worker automatically runs `quality.pipeline.run_panel`
synchronously after committing a successful metadata-mode quality-v1 formal
review. It creates the empty review page, scores the retained 1024 images with
TOPIQ, and assembles PRDC and unreviewed evidence. No second evaluation queue or
GPU process is created. `quality-v1/status.json` records panel errors separately;
formal `summary.json` and its completed journal request remain intact. Missing
manual labels remain `REVIEW_REQUIRED`. Historical objectives are skipped.

New automatic `protocol.json` records actual sampling/precision/noise settings,
ordered reference identities, and feature extractor settings. Its canonical JSON
content is the protocol identity; this does not compute a file hash. Historical
baselines have short manually registered protocol names. Before comparing them
to a new review, inspect their original summary and existing generation/training
records, reconstruct the actual descriptor, and verify all relevant settings
match. Do not assign the same identifier just to pass the gate. Missing old
precision or sampling evidence stays unresolved until this metadata comparison
is completed; no image regeneration is implied or authorized.
