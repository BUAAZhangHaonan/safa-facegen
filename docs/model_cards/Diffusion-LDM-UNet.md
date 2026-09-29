# Diffusion-LDM-UNet — v1.0-pretrained

## 权重与接口

检查点：`Diffusion-LDM-UNet-0741ep-20260924T143627Z`  
角色：EMA；阶段内步数 147178；HQ 曝光 741.48128 轮。  
选中权重训练目标：`epsilon_mse`。  
输出：无条件 256×256 RGB，float32，范围 [-1,1]。  
输入噪声形状：`[3, 64, 64]`；额外逐步噪声数：200。  
配套 codec：`ldm_vq4`。具体采样参数随 `release.json` 保存。

## 已有画质记录

FID1024：33.887628；KID：0.00265398。真实与生成样本各1,024张。  
协议：`diffusion_741_special_fp32_tf32_off`。历史精度字段：`documented`。  
默认生成入口；父候选固定256张 AI 标签为 acceptable197、minor59、severe0、uncertain0，细节软化和局部纹理持续。

固定原图、AI 审阅标签和原始评价协议继续作为画质附件。AI 标签的审阅者、uncertain 项和原始计数保持原记录。

## 使用定位

用于下游 SAFA 的冻结生成器和可微噪声输入，也用于无条件人脸生成的预训练初始化。`.generate(num_images=1, seed=42)` 提供空条件调用，`.sample(..., grad_enabled=True)` 提供噪声梯度调用。

## 验收与来源

软件验收结果由 K100 的 `smoke_release.py` 实际执行后写入 `release-smoke.json`。来源依据为仓库资产清单和2026年9月29日总结第9节。上游代码、初始化权重、codec 与数据许可分别登记在 `THIRD_PARTY_NOTICES_zh.md` 及项目发布记录。

固定10K专项：FID 8.9103761611，KID 0.003036061369；DDIM200、eta1、FP32、TF32关闭。
