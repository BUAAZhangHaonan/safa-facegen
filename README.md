# safa-facegen

六种无条件人脸生成器的训练、权重管理和 PyTorch 可微生成接口。输出分辨率为256×256，训练数据为 FFHQ 70,000张与 CelebA-HQ 30,000张 JPEG95 图片。

| 模型 | 训练实现 | 初始化 | 采样 |
|---|---|---|---|
| MeanFlow-B-4 | 原作者 JAX DiT | 迁移已有1751轮权重 | 单步 |
| MeanFlow-B-2 | 原作者 JAX DiT | 迁移已有240轮权重 | 单步 |
| MeanFlow-L-2 | 原作者 JAX DiT | 迁移已有397轮权重 | 单步 |
| RectifiedFlow-NCSNpp | 原作者 NCSN++ | CelebA-HQ 官方权重 | Dormand–Prince ODE |
| Diffusion-LDM-UNet | 原作者 FFHQ LDM-VQ-4 | FFHQ 官方权重 | 200步 DDIM |
| LatentConsistency-LDM-UNet | LDM骨干与官方一致性蒸馏公式 | 本项目经过人工验收的 Diffusion | 4步 |

模型的质量状态以 `models/formal/` 中的人工验收记录为依据。实现验证、训练启动和正式质量验收分别记录。

## 项目结构

- `src/safa_facegen/meanflow`：原作者 JAX 训练、旧权重转换和 PyTorch 导出。
- `src/safa_facegen/torch_models`：RF、LDM、LCM 骨干、目标函数和可微采样。
- `src/safa_facegen/data.py`、`cache.py`：数据清单、确定性增强和两种潜变量缓存。
- `src/safa_facegen/controller.py`、`replicate.py`：四卡训练队列、资源保护、跨机存储与评价。
- `configs`：六个模型的训练配方和队列配置。
- `vendor`：按固定提交保存的必要官方源码、许可证及修改说明。
- `models`、`runs`、`data`、`reports`：机器本地资产和运行记录，通过 Git 忽略。

## 机器角色

H100 在现有 `meanflow_e15_h100_bundle` 根目录内执行全部训练。K100 使用 `safa-facegen` 保存检查点、进行生成评价并向独立 SAFA 研究项目提供生成器。两个代码副本使用同一 Git 提交。

项目内 `.venv-torch` 和 `.venv-jax` 分别管理 PyTorch 与 JAX。`requirements/` 保存各机器环境的实际依赖版本；CUDA环境通过真实计算和四卡恢复测试验收。机器专用路径写入忽略的 `configs/local.json`。H100 不安装代理工具，外部依赖通过获准的 SSH 链路取得；训练进程使用项目内缓存并关闭在线模型下载。

```bash
export PYTHONPATH="$PWD/src"
export SAFA_FACEGEN_ROOT="$PWD"
.venv-torch/bin/python -m safa_facegen.cli status
.venv-torch/bin/python -m safa_facegen.controller --campaign configs/campaign.json
```

控制器需要对应模型已经通过训练恢复验证和batch标定。它每次仅启动一个四卡任务，顺序为 Diffusion、LCM、RF、MeanFlow B/4、B/2、L/2。正式完成通过人工图像验收决定；等待审核期间继续训练当前模型。

## 检查点与评价

检查点名称使用 `模型名称-本次HQ阶段完成轮数-UTC时间`。元数据额外保存精确优化器步数、有效样本曝光数、数据与编解码器哈希、代码提交、初始化来源和异常恢复记录。历史数据阶段的轮数单独登记。

每15分钟保存完整恢复状态，每30分钟输出64张预览，每2小时提交1024张样本的评价，并展示其中固定前256张。记录单脸检测、空白或非有限图、FID-1024和KID；样本不按质量筛选。FID-1024的样本预算在指标名称中明确保留。

训练恢复状态与正式 EMA 使用不同角色标识。跨机传输使用临时文件与内容校验，正式评价只接受明确的 EMA 导出。

## PyTorch 接口

```python
import torch
from safa_facegen import load_generator

generator = load_generator(model_id, checkpoint)
noise = torch.randn(1, *generator.noise_shape, device="cuda", requires_grad=True)
images = generator.sample(noise, step_noises=step_noises, grad_enabled=True)
```

`images` 为 `[B,3,256,256]` RGB Tensor，值域约定为 `[-1,1]`。生成器参数保持冻结，输入噪声可以接收梯度。随机多步采样通过显式 `step_noises` 复现；模型暴露 `step_noise_count` 和 `step_noise_shape`。

两参数加载会读取项目 `configs/local.json` 中该模型登记的 `paths.codec`；未设置覆盖时，MeanFlow 使用 `models/codecs/SD-VAE-EMA`，Diffusion 与 LCM 使用 `models/codecs/LDM-VQ4.pt`。路径按 `SAFA_FACEGEN_ROOT` 或当前安装源码的项目根目录解析，缺失路径会直接报错。显式 `codec=` 参数可以指定已登记的编解码器，加载过程继续核对权重中的编解码器身份、结构和缩放。

LDM与LCM保留官方VQ量化的直通梯度估计。预训练始终使用无条件输入，SAFA 的条件注入在独立研究项目中实现。

## 数据与来源

图片及其缓存仅存储在计算服务器。FFHQ、CelebA-HQ、各官方权重和源代码分别保留来源及许可说明；这些资产不通过代码仓库分发。详见 `docs/data.md`、各 `vendor` 目录中的许可证，以及 `docs/initializations.json`。
