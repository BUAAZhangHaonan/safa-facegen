# MeanFlow 实现

三种 MeanFlow 使用 [Gsunshine/meanflow 固定提交](https://github.com/Gsunshine/meanflow/tree/d70cb55d298ee03c53bf6da67bec281082e4e2d9) 的 JAX DiT 主干、NULL 类索引 1000、SD-VAE-EMA 和一步采样。固定源码及 MIT 许可位于 `vendor/meanflow`；本地相对导入和 JAX API 调整记录在 `vendor/meanflow/PATCHES.md`。

B/4、B/2、L/2 的迁移来源分别为已完成的 1751、240、397 轮权重，精确来源和权重角色见 [initializations.json](initializations.json)。B/2 后续完成 iMF 边界速度适配；B/4 与 L/2 保留各自选中 EMA。画质比较与当前选择见 [results.md](results.md)。

## 训练目标与配方

原始 MeanFlow 目标使用官方 FP32 JVP、logit-normal 时间采样、75% 边界样本、自适应权重以及四卡 pmap。B/2 适配在现有主干上使用 iMF 边界速度目标：边界与区间样本各占一半，时间分布为 logit-normal μ=-0.4、σ=1，归一化 `p=1`、`eps=0.01`。公式实现位于 `src/safa_facegen/meanflow/boundary.py`。

| 阶段 | 更新数 | 每卡 batch | 峰值→末端学习率 | EMA | 精度 |
| --- | ---: | ---: | --- | ---: | --- |
| B/2 iMF 边界速度 | 20,000 | 64 | 3e-5 → 3e-6 | 0.999 | FP32 / JAX matmul highest |

学习率预热 500 步后按 cosine 下降；Adam β=(0.9,0.95)，weight decay=0。已执行的阶段配方保存在 `configs/quality/meanflow-b2.json` 与同目录单模型 campaign，阶段登记和完整训练状态保存原始阶段身份。旧目标与 B/4、L/2 的基础配方分别保存在 `configs/meanflow-b4.json`、`meanflow-b2.json`、`meanflow-l2.json`。训练历史见 [training-history.json](training-history.json)。

训练使用 100k 图对应的只读 MeanFlow 潜变量缓存。每个样本的原图与翻转图分别编码，缓存 posterior mean/std；训练时抽取 posterior 样本并乘 0.18215。图片翻转与样本顺序见 [data.md](data.md)。

## 权重与推理

`meanflow/convert.py` 提供迁移时的严格键名与形状转换，也承担训练器当前的 canonical/Flax 转换和 EMA 导出。训练状态保存 raw、EMA、优化器、随机状态、样本位置、配方及阶段身份；导出目录保存可由 PyTorch 严格加载的 EMA safetensors 和 manifest。具体保留身份见 [artifacts.json](artifacts.json)。

```python
from safa_facegen import load_generator

generator = load_generator("MeanFlow-B-2", checkpoint, codec=codec_path)
rgb = generator.sample(noise, grad_enabled=True)
```

`noise_shape=(4,32,32)`，输出为 `[B,3,256,256]` RGB `[-1,1]`。生成器参数冻结，输入 noise 保留梯度。推理使用一步 `noise-u(noise,t=1,h=1)`；EMA、codec 及架构身份由保存 manifest 核对。训练环境依赖见 `requirements-jax.txt` 和 `requirements/h100-jax.lock`。
