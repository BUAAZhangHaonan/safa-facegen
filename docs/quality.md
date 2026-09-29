# 画质评价与选择

本轮四项有限适配已经完成训练、正式评价和最终副本同步。每项使用原登记的阶段配方及独立 EMA 身份；实际指标、AI审阅标注与当前选择见 [results.md](results.md)，对应权重和报告见 [artifacts.json](artifacts.json)。

| 模型 | 方法 | 固定更新数 | 评价状态 |
| --- | --- | ---: | --- |
| MeanFlow-B-2 | iMF 边界速度 | 20,000 | 正式评价完成 |
| Diffusion-LDM-UNet | Min-SNR epsilon | 15,000 | 正式评价完成 |
| RectifiedFlow-NCSNpp | 四卡当前批次全局 OT | 10,000 | 正式评价完成 |
| LatentConsistency-LDM-UNet | 真实数据一致性细化 | 10,000 | 正式评价完成 |

B/4 与 L/2 保留原选中 EMA；B/2 的本轮画质比较未触发这两个模型的适配。Diffusion 的完整比较进入 LCF 路线，LCF 使用原选中 LCM EMA 和真实数据。执行配方见 `configs/quality/`，原始阶段 ID、目标 ID 和选择证据保存在登记与训练历史中。

## 评价材料

每个候选保留完整的 1,024 张原始生成图，索引前 256 张用于原分辨率结构审核。正式评价记录 FID1024、KID、单脸检测、空白及非有限图；TOPIQ-NR 记录纹理清晰度，PRDC coverage 记录分布覆盖。AI审阅标签逐图记录 `acceptable`、`minor`、`severe`、`uncertain`，并分别记录纹理和背景人脸。自然表情、牙齿及遮挡按原图语境审核。

比较核对模型、采样与精度、固定参考图顺序和特征提取器。候选与基线的协议描述保存在 `protocol.json`；审核材料包括署名标签、纹理比较和候选证据，历史协议缺项随比较结果记录。选择结果使用 `BETTER`、`NOT_BETTER` 或 `REVIEW_REQUIRED`。最终验收由用户决定。数据与采样规则见 [data.md](data.md)。

K100 的正式审核目录为 `reports/evaluation/<checkpoint_id>/review/`，质量材料位于其 `quality/` 子目录。独立的 [六模型原图联系表](../reports/final/gallery.html) 汇集六个选中 EMA 的原图。`src/safa_facegen/quality/README.md` 记录证据工具的实际命令。
