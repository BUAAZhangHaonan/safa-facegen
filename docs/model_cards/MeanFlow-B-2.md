# MeanFlow-B-2 — v1.0-pretrained

## 权重与接口

检查点：`MeanFlow-B-2-0679ep-20260926T041357192035Z`  
角色：EMA；阶段内步数 66599；HQ 曝光 679.58368 轮。  
选中权重训练目标：`original_meanflow`。  
输出：无条件 256×256 RGB，float32，范围 [-1,1]。  
输入噪声形状：`[4, 32, 32]`；额外逐步噪声数：0。  
配套 codec：`sd_vae_ema`。具体采样参数随 `release.json` 保存。

## 已有画质记录

FID1024：38.04357；KID：0.00569139。真实与生成样本各1,024张。  
协议：`historical_recorded_protocol`。历史精度字段：`partially_recorded`。  
一步主力；父候选 AI 标签为 acceptable169、minor57、severe5、uncertain25。

固定原图、AI 审阅标签和原始评价协议继续作为画质附件。AI 标签的审阅者、uncertain 项和原始计数保持原记录。

## 使用定位

用于下游 SAFA 的冻结生成器和可微噪声输入，也用于无条件人脸生成的预训练初始化。`.generate(num_images=1, seed=42)` 提供空条件调用，`.sample(..., grad_enabled=True)` 提供噪声梯度调用。

## 验收与来源

软件验收结果由 K100 的 `smoke_release.py` 实际执行后写入 `release-smoke.json`。来源依据为仓库资产清单和2026年9月29日总结第9节。上游代码、初始化权重、codec 与数据许可分别登记在 `THIRD_PARTY_NOTICES_zh.md` 及项目发布记录。
