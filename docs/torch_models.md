# PyTorch 模型

Rectified Flow、Diffusion 和 Latent Consistency 共用 `src/safa_facegen/torch_models` 的冻结可微生成接口。RF 在 RGB 空间运行；后两者使用 FFHQ LDM-VQ4 的 `[3,64,64]` prequant latent。固定上游源码、提交和许可证保存在 `vendor`，初始化权重来源见 [initializations.json](initializations.json)。

| 模型 | 骨干与来源 | 当前目标 | 采样 |
| --- | --- | --- | --- |
| RectifiedFlow-NCSNpp | [RectifiedFlow NCSN++](https://github.com/gnobitab/RectifiedFlow/tree/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6) | 四卡当前批次全局 OT | 可微 RK45，容差 1e-5 |
| Diffusion-LDM-UNet | [CompVis FFHQ LDM](https://github.com/CompVis/latent-diffusion/tree/a506df5756472e2ebaf9078affdde2c4f1502cd4) | Min-SNR epsilon，γ=5 | DDIM 200 步，η=1 |
| LatentConsistency-LDM-UNet | [LCM 蒸馏公式](https://github.com/luosiallen/latent-consistency-model/tree/a9ad79587cc8bd1e404ccd1a3056a3da969b2f62) 与项目 FFHQ UNet | 真实数据一致性细化，skip=20 | 4 步 |

LCM 的原始学生由已批准的项目 Diffusion EMA 初始化。当前 LCF 阶段从已选 LCM EMA 初始化，使用真实数据和 FP32 EMA 目标；采样保持四步。其固定推理时间点为 `[999,759,499,259]`。

## 已执行适配

| 阶段 | 更新数 | 每卡 batch | 峰值→末端学习率 | EMA | 精度 |
| --- | ---: | ---: | --- | ---: | --- |
| Diffusion Min-SNR | 15,000 | 64 | 3e-6 → 3e-7 | 0.999 | BF16 前向、FP32 参数 |
| RF 全局当前批次 OT | 10,000 | 12 | 3e-6 → 3e-7 | 0.999 | FP32 |
| LCF 真实数据细化 | 10,000 | 64 | 2e-6 → 2e-7 | 0.9999 | BF16 前向、FP32 参数 |

三个阶段均使用四卡、500 步预热、cosine 学习率、Adam β=(0.9,0.999)、weight decay=0。OT 将四卡当前 batch 聚合后执行一次一对一分配，上限为 48 个样本。完整配方与原始阶段身份保存在 `configs/quality/` 的三个单模型配置和 campaign；训练结果见 [results.md](results.md)。

训练状态保存模型、EMA、优化器、随机状态、数据游标、配方和阶段身份。完整恢复和独立 EMA 导出各自使用固定格式身份。阶段、目标和格式标识的原始序列化值由 `src/safa_facegen/contracts.py` 固定。保留的恢复状态及六个选中 EMA 见 [artifacts.json](artifacts.json)。

## 生成器接口

```python
from safa_facegen import load_generator

generator = load_generator(model_id, ema_checkpoint, codec=codec_checkpoint)
rgb = generator.sample(noise, step_noises=step_noises, grad_enabled=True)
```

RF 的 `noise_shape=(3,256,256)`；Diffusion 与 LCM 为 `(3,64,64)`。`step_noise_count` 和 `step_noise_shape` 描述多步采样所需的显式噪声。参数保持冻结，输出 `[B,3,256,256]` RGB `[-1,1]`；输入与逐步噪声的梯度经过执行的采样路径传播。LDM/LCM 解码采用官方 VQ 量化直通梯度，RF 求解器采用 FP64 状态和 FP32 神经速度场。

正式评价读取独立 EMA 和已登记 codec 身份，使用 [data.md](data.md) 的固定输入与参考集。来源文件及许可证说明见 `vendor/latent_diffusion`、`vendor/latent_consistency` 和 `vendor/rectified_flow`；RF 的 SciPy RK45 许可保存在 `vendor/rectified_flow/SCIPY_LICENSE.txt`。
