# SAFA FaceGen v1.0-pretrained

六种无条件 256×256 RGB 人脸预训练生成器，配套冻结参数的生成接口和可微噪声输入。默认入口为 Diffusion-LDM-UNet，四步入口为 LatentConsistency-LDM-UNet。

| 模型 | 发布 EMA | 该权重的训练目标 | 采样 |
|---|---|---|---|
| MeanFlow-B-4 | 1908.4096 HQ轮 | 原始 MeanFlow | 一步 |
| MeanFlow-B-2 | 679.58368 HQ轮 | 原始 MeanFlow | 一步 |
| MeanFlow-L-2 | 249.3072 HQ轮 | 原始 MeanFlow | 一步 |
| Diffusion-LDM-UNet | 741.48128 HQ轮 | epsilon MSE | DDIM200，eta1 |
| RectifiedFlow-NCSNpp | 7.0144 HQ轮 | 独立噪声配对的 RF 速度目标 | RK45，atol/rtol=1e-5 |
| LatentConsistency-LDM-UNet | 19732步 | 教师741轮的 LCD | 四步，999/759/499/259 |

## 空条件生成

使用项目保存的 K100 PyTorch 运行环境；依赖版本见 `requirements/k100-torch.lock`。完整解压目录中保留 `src/` 与 `vendor/`。

```bash
python generate.py --device cuda:0 --output face.png
python generate.py --model LatentConsistency-LDM-UNet --output face-lcm.png
```

```python
from pathlib import Path
from runtime import load_selected

generator, metadata = load_selected(Path('.'), device='cuda:0')
images = generator.generate(num_images=1, seed=42)
```

`images` 为 `[1,3,256,256]` RGB Tensor，范围 `[-1,1]`。该简便入口在运行设备上产生随机噪声；正式历史评价采用各自记录的 CPU 逐样本噪声协议。

## SAFA 噪声梯度接口

```python
import torch
noise = torch.randn(1, *generator.noise_shape, device='cuda:0', requires_grad=True)
extra = [torch.randn_like(noise, requires_grad=True)
         for _ in range(generator.step_noise_count)]
images = generator.sample(noise, step_noises=extra, grad_enabled=True)
```

模型参数冻结。Diffusion 与 LCM 的 VQ 解码采用原生量化及 straight-through 梯度；RF 使用像素空间 ODE；MeanFlow 使用 SD-VAE-EMA。

## 画质档案

六张模型卡列出对应 EMA 的已有指标和局部缺陷记录。图像仍可出现纹理覆盖、软化、肤色偏差及局部五官问题。固定原图与 AI 审阅标签按检查点保存，uncertain 项保持原标注。正式发布保存这些质量特征，供下游选用。

四条后续适配合计55,000更新，结果保存在项目历史中。本版本选用上表六份核心 EMA。TOPIQ 作为纹理辅助字段保留；预训练交付验收检查来源、可加载性、生成接口和噪声梯度。

## 文件与许可

`release.json` 是权重、codec、采样参数和模型卡的入口。模型文件使用原有 EMA 格式。`release-smoke.json` 保存本目录的实际接口验收结果。

本地封版由项目所有者执行。公开分发时随文件提供 `THIRD_PARTY_NOTICES_zh.md` 及逐项核对后的许可记录。数据图像、训练缓存和优化器状态保存在项目私有恢复档案中。
