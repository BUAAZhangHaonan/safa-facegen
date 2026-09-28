# Quality-v1 implementation and execution contract

The 2026-09-28 owner request authorizes a finite adaptation plan. Start with
MeanFlow-B/2, adapting the iMF boundary-velocity objective on the existing model.
This is not an implementation of the full iMF architecture. The backbone,
NULL1000, one-step inference, cached data, SD-VAE and 0.18215 scale stay fixed.

## First production stage

Use the existing controller with
`configs/quality-v1/meanflow-b2-campaign.json`. Parent EMA is
`MeanFlow-B-2-0679ep-20260926T041357192035Z` (66599 previous-stage updates).
Initialize both raw and EMA from that EMA; start fresh Adam, RNG/cursor and
stage step zero. The new absolute stop is 20000, microbatch 64 on each of four
GPUs, FP32 with JAX matmul precision highest, EMA 0.999, Adam (0.9, 0.95),
zero weight decay. LR warms up for 500 updates to 3e-5, then cosine decays
to 3e-6. Interrupted execution restores this stage's full state.

Registrations use `runs/controller/quality-v1/<model>.bounded-stage.json`;
legacy registrations remain intact. Each model has exactly one new slot;
both alternative LCM objectives share its slot. Stage-specific latest pointers
prevent reading a previous recipe's optimizer. The registry fixes the parent,
objective, recipe, zero origin and absolute stop. Recovery can only reduce a
persisted multiplier on the schedule; its peak is not confused with the current
warmup rate. The original controller, lock and resource rules remain authoritative.

## Other objectives and conditional execution

`configs/quality-v1/production-plan.json` is a plan specification, not a native
campaign. The native training entries support Min-SNR epsilon weighting,
global-batch RF OT, teacher LCD and real-data LCF. New native campaigns must
resolve real parent artifacts before execution. Native `diffusion-campaign.json`
and `rectified_flow-campaign.json` are prepared with their retained 741/7 EMA
paths; preparation does not register or launch those slots. No other model starts as a
side effect of completing B/2.

B/4 and L/2 require an applicable B/2 quality comparison returning BETTER.
New-teacher LCD requires a selected Min-SNR Diffusion candidate. LCF is mutually
exclusive and requires complete Diffusion-stage evidence returning NOT_BETTER;
missing labels or unfinished evidence cannot select LCF. Conditional evidence
is rechecked when binding the immutable registration. Exact fixed budgets and
recipes are enforced in `quality_stage.py`; there is no automatic second round.

## Integrity and sampling

New stages use metadata integrity: real filenames, nonzero byte sizes, loadable
strict model state, codec/model identity and atomic manifest/completion records.
Save, restore, export, replica and evaluator support this contract without new
whole-file SHA256 scans. Historical hash strings can remain provenance fields;
they are not invented for new exports. Existing deterministic augmentation
BLAKE2 is untouched. Partial transfers and their existing interval are preserved.

Both generator families expose `generate(num_images=1, seed=None)`, returning
RGB tensors in [-1,1]. A supplied integer seeds a private generator including
all stochastic sampler steps. Existing differentiable `sample(noise, ...)`
remains the SAFA integration API. Inference contains no image repair or filtering.

## Quality decisions

See `src/safa_facegen/quality/README.md`. Reuse all 1024 original generated images
and existing per-image Inception features. TOPIQ-NR measures texture/clarity,
PRDC coverage detects distribution narrowing, and the unchanged first 256
images require explicit structural labels. Natural expression, teeth and
occlusion are not automatically defects. Background faces are recorded separately.

The empty-label artifact intentionally yields REVIEW_REQUIRED. TOPIQ cannot fill
structural labels or approve a model. A signed original-resolution texture
comparison can support practical improvement; thresholds require fixed manually
reviewed anchors and are not refitted per checkpoint. The release gate can only
return READY_FOR_OWNER_ACCEPTANCE with quality_approved=false. Owner acceptance,
stage completion and candidate selection are separate facts.

Software validation and isolated small-batch GPU checks are not training-history
entries and are not evidence of improved image quality. Formal stage facts go in
`docs/training-history.json`; images, weights and diagnostic artifacts stay out
of Git. No extra seeds, parameter grids or repeated historical evaluations are
part of this plan.
