# safa-facegen

六种无条件人脸生成器的训练、评价与 PyTorch 可微生成接口。输出为 256×256 RGB；训练数据由 FFHQ 70,000 张和 CelebA-HQ 30,000 张组成。

| 模型 | 实现目标 | 采样 |
| --- | --- | --- |
| MeanFlow-B-4 | 原始 MeanFlow | 单步 |
| MeanFlow-B-2 | iMF 边界速度 | 单步 |
| MeanFlow-L-2 | 原始 MeanFlow | 单步 |
| RectifiedFlow-NCSNpp | 四卡当前批次全局 OT | 自适应 RK45 |
| Diffusion-LDM-UNet | Min-SNR epsilon | 200 步 DDIM |
| LatentConsistency-LDM-UNet | 真实数据一致性细化 | 4 步 |

B/2、Diffusion、Rectified Flow 和 Latent Consistency 的四项适配共 55,000 次更新，训练、正式评价和最终副本同步均已完成。B/4 与 L/2 保留各自选中的 EMA，未进入本轮适配。H100 当前没有训练作业；K100 的空闲复制与评价 worker 已于 2026-09-29 06:00 UTC 停止。六个选中 EMA 与各模型最新完整恢复状态共 11 个 checkpoint 身份纳入保留清单，中间权重已完成清理。

最终结果见 [结果与选择](docs/results.md)、[资产清单](docs/artifacts.json) 和 [六模型原图联系表](reports/final/gallery.html)。图像与AI审阅标注、原始初始化权重、codec、10 万张训练图和两类潜变量缓存保存在计算服务器。

## 使用接口

```python
import torch
from safa_facegen import load_generator

generator = load_generator(model_id, checkpoint)
noise = torch.randn(1, *generator.noise_shape, device="cuda", requires_grad=True)
step_noises = [
    torch.randn(1, *generator.step_noise_shape, device="cuda")
    for _ in range(generator.step_noise_count)
]
images = generator.sample(noise, step_noises=step_noises, grad_enabled=True)
```

`images` 的形状为 `[B,3,256,256]`，值域为 `[-1,1]`。生成器参数冻结，输入噪声与显式逐步噪声可参与梯度计算。MeanFlow 使用 SD-VAE-EMA，Diffusion 与 Latent Consistency 使用 LDM-VQ4；`load_generator` 从项目本机配置或显式 `codec=` 参数解析已登记的 codec。

## 代码与资料

`src/safa_facegen/meanflow` 保存 JAX 训练、严格权重转换和 PyTorch 推理；`src/safa_facegen/torch_models` 保存 RF、Diffusion 与 Latent Consistency 的骨干和目标函数。`controller.py` 读取显式指定的单模型 campaign；`replicate.py` 管理跨机副本和正式评价。`configs/quality` 保存本轮已执行阶段的原始配方和登记身份；阶段 ID、目标 ID 及 checkpoint 格式值保留在保存记录中。`vendor` 保存固定上游提交、许可及修改说明。

实现与协议分别见 [数据](docs/data.md)、[MeanFlow](docs/meanflow.md)、[PyTorch 模型](docs/torch_models.md)、[跨机存储](docs/storage.md) 和 [画质证据](docs/quality.md)。初始化来源见 [initializations.json](docs/initializations.json)，完整阶段记录见 [training-history.json](docs/training-history.json)。

训练图像、模型权重和运行记录由服务器本地管理；Git 仓库保存源码、配置和来源说明。
