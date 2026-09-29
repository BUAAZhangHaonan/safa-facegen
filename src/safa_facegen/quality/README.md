# 画质证据工具

K100 的正式 `review` 目录提供 1,024 张原图、逐图 Inception 特征和评价摘要。`quality.pipeline.run_panel` 在正式评价完成后写入 `review/quality/`：原分辨率审核页、TOPIQ-NR、PRDC coverage、协议描述和待人工标注的证据。人工标注与选择流程见 [画质评价与选择](../../../docs/quality.md)。

独立运行入口：

```bash
python -m safa_facegen.quality.review prepare --review "$REVIEW" --out "$REVIEW/quality"
python -m safa_facegen.quality.review score --review "$REVIEW" --out "$REVIEW/quality" --device cuda:0 --batch 16 --weights models/evaluation/cfanet_nr_koniq_res50-9a73138b.pth
python -m safa_facegen.quality.evidence --review "$REVIEW" --out "$REVIEW/quality/candidate.json" --protocol-id "$REGISTERED_PROTOCOL"
```

完成原分辨率逐图审核后，使用签名标签和纹理比较装配证据：

```bash
python -m safa_facegen.quality.evidence --review "$REVIEW" --out "$REVIEW/quality/candidate-reviewed.json" --protocol-id "$REGISTERED_PROTOCOL" --labels "$REVIEW/quality/labels-reviewed.csv" --texture-review acceptable --texture-reviewer "$REVIEWER" --texture-comparison "$SIGNED_TEXTURE_COMPARISON_JSON" --baseline "$BASELINE_REVIEWED_JSON"
python -m safa_facegen.quality.gate "$REVIEW/quality/candidate-reviewed.json" --baseline "$BASELINE_REVIEWED_JSON" --select-better
```

`--texture-comparison` 登记原分辨率审核结论、审核人、基线 checkpoint 身份和依据。结构标签覆盖固定前 256 张，纹理及背景人脸分别记录。选择输出为 `BETTER`、`NOT_BETTER` 或 `REVIEW_REQUIRED`；协议、标签和原图保存在对应正式报告目录。最终候选见 [results.md](../../../docs/results.md)。
