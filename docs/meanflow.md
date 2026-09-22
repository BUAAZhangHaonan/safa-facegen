# MeanFlow 实现与操作

主干和原始 MeanFlow 目标来自 `Gsunshine/meanflow` 提交
`d70cb55d298ee03c53bf6da67bec281082e4e2d9`，MIT 许可，见
`vendor/meanflow/UPSTREAM.json`、`LICENSE`、`PATCHES.md`。只调整包相对导入
及 JAX 的 clip API。

## 环境与入口

训练环境使用 `requirements-jax.txt`，Python 3.11/3.12，Linux，NVIDIA 驱动
至少 580。它与 PyTorch 环境分开。转换和 PyTorch 推理需要项目 PyTorch
环境中的 torch、safetensors、diffusers；训练进程不导入 torch。

H100 的 JAX venv 复用现有 CUDA13 包；启动时需要把现有环境的
`nvidia/cu13/lib`、`nvidia/cudnn/lib`、`nvidia/nccl/lib` 加入 `LD_LIBRARY_PATH`。
这些目录来自 `/opt/conda/private/envs/meanflow/lib/python3.12/site-packages/`。
部署控制器负责传入运行环境，不安装网络代理软件。

在仓库根目录执行并设置 `PYTHONPATH=src`，所有相对路径以当前目录为基准。
`SAFA_MEANFLOW_VENDOR` 可指定 vendor 位置；默认使用仓库内固定源码。

```bash
python -m safa_facegen.meanflow.convert \
  --checkpoint models/initialization/MeanFlow-B-4-1751ep-20260701T114714Z.pt \
  --model-id MeanFlow-B-4 \
  --output models/initialization/MeanFlow-B-4-0000ep-<实际UTC时间>

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m safa_facegen.meanflow.trainer \
  --config configs/meanflow-b4.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m safa_facegen.meanflow.trainer \
  --config configs/meanflow-b4.json --resume latest
```

示例旧文件名仅展示参数形式；实际使用注册表里的原文件路径。实施者将配置中
`paths.initial_checkpoint` 设为转换生成的目录。B/2、L/2 对应配置已提供。
`--max-steps` 用于有限步性能/恢复验证，也会写完整检查点；`--microbatch`
用于明确指定每设备 batch。续训默认使用保存的 microbatch，包括已记录的恢复调整。

## 严格转换与推理

转换要求同时存在 `model_state_dict` 与 `ema_model_state_dict`，可通过显式
`--raw-key/--ema-key` 指定其他已核验字段。拒绝缺失、多余、维度错误、NaN/Inf
参数。转换程序是唯一读取旧 pickle 的位置，限可信项目检查点。训练和推理只读
新 safetensors。Torch Linear `[out,in]` 转 Flax `[in,out]`，卷积 OIHW 转 HWIO，
QKV 拼接顺序不变；逐个 key 校验实际官方 Flax 初始化树。

旧固定 NULL 的 zMLP 折叠为官方 NULL embedding 行 1000。raw/EMA 独立折叠。
残差 hidden 半区置换同时作用于 patch、attention/MLP 输入输出及 adaLN，保留
旧模型函数并采用官方 `[x,y]` 位置编码。旧优化器不复用；HQ 新阶段从 0 开始，
原 epoch/step/文件哈希写在来源记录。后续训练完整恢复 JAX 优化器。

```python
from safa_facegen.meanflow import MeanFlowGenerator
g = MeanFlowGenerator.from_pretrained(export_directory, codec=vae_directory, device="cuda")
noise.requires_grad_(True)
rgb = g.sample(noise, grad_enabled=True)  # [N,3,256,256], [-1,1]
```

`noise_shape=(4,32,32)`，`step_noise_count=0`，`state_role='ema'`。
`ema_sha256` 为单独 EMA safetensors 的真实哈希。
正式加载只接受专用 `ema.safetensors`，核对请求模型、官方提交、NULL=1000 和
latent scale=0.18215。codec 路径旁须有同名 `.json` 登记文件；加载复用登记身份
与文件 stat，并与训练 cache 的 codec 身份核对，不重新扫描权重内容。
所有参数冻结，`grad_enabled=True` 保留 noise→网络→VAE 的梯度；默认无梯度。VAE 本地路径显式传入，
不隐式下载或切换 codec。原 MeanFlow 只执行一步 `noise-u(noise,t=1,h=1)`。

## 数据与训练

缓存为只读 float32 NumPy memmap `[N,2,8,32,32]`：原图/水平翻转图分别 VAE
编码，前四通道 mean、后四通道 std，均未乘 scale。每次训练重新采样 posterior
并乘 0.18215。MeanFlow 继续使用 `sd-vae-ft-ema`。训练拒绝 cache 与 dataset
manifest 的已登记大小/mtime 不匹配。复用缓存构建时登记的数据身份和样本数，
训练启动不重读 100k 条记录。训练时检查 batch 有限值和非负 std。

shuffle 为 `(seed,sampler_epoch)` 确定的 permutation；flip 使用统一
`BLAKE2b(f'{seed}:{epoch}:{index}',digest_size=1)[0]&1`。每条数据每轮访问一次，
尾批缩小，数据总量必须整除设备数，无填充样本。保存 epoch、cursor 和样本计数。

一进程 pmap 四张卡；官方 FP32 JVP/目标不变。lr=1e-4，Adam β=(0.9,0.95)，
eps=1e-8，weight decay=0，EMA=0.9999；logit-normal μ=-0.4/σ=1，75% r=t，
adaptive norm p=1/eps=0.01，固定 NULL，关闭 guidance 和 class dropout。
microbatch 是每设备图像数；B/4、B/2、L/2 分别使用 1280、256、96，四卡有效 batch 分别为 5120、1024、384。学习率、Adam 参数和 EMA 系数保持上述配方。
三份配置显式记录上述固定配方，包括 fp32、Adam eps、NULL index、latent scale
及 class dropout。启动时与实现使用的固定值逐项核对；缺少字段或改变固定配方会失败。
完整有效配置随每个 checkpoint 保存，学习率的自动调整另记录在状态和元数据中。
XLA 持久编译缓存存放在项目 `.cache/jax`，相同模型和 batch 的重启可复用编译结果。

## 保存、资源和评价

Orbax 保存 raw、EMA、Adam moments/count、步数、JAX RNG 和学习率。元数据保存
数据/cache/codec/初始权重身份、NumPy RNG、shuffle epoch/cursor、attempted/
valid samples、OOM/无效步数、当前 batch、周期时刻。恢复时严格核对身份。
先从设备去除四卡复制轴，再复制到主存。同步保存全部分片后写 COMPLETE，并
原子重命名目录；manifest 记录 state 下每个文件的 SHA256 和大小。只恢复 COMPLETE
可见且 manifest、完整 state 文件清单与逐文件校验均通过的检查点；已经由保留策略写入
`STATE_RETIRED.json` 的旧目录不能恢复。

名称为 `MeanFlow-B-4-0000ep-<真实UTC时间>` 等，轮数只计当前 HQ 阶段有效样本，
旧数据阶段训练量保存在 provenance。不会把训练未完成的轮数加到名称里。

默认 JAX 预分配 85%；RAM soft/hard=192/224 GiB，GPU allocator peak 上限 72 GiB。
达到 RAM soft/hard 阈值时释放当前进程可释放缓存并停止，保留最近完整检查点。
所有保存入口都先检查内存，包含退出、信号、标定与自动恢复路径；超过阈值不创建
新的 CPU 状态副本。显存超预算记录并减小恢复 batch 后退出。OOM/NaN 最多连续
自动恢复两次：OOM 调小 batch；NaN 保留 batch、lr 减半并维持 fp32。失败步不更新
参数、EMA、RNG、有效样本；耗尽后非零退出，内存允许时保存最后有效状态。
控制器明确提供的恢复 batch/LR 在恢复完整检查点后生效，事件记录来源与前后值；
该 trainer 使用直接 memmap 读取，拒绝无效的 worker/prefetch 恢复参数。

每 900 秒保存；每 1800/7200 秒导出 EMA 并向 `requests.jsonl` 写 preview/review
请求，包含 codec、seed、step 对应检查点。K100 消费这些请求执行真实解码和评价。
训练中不加载 VAE/Inception GPU，不把 50k RGB 评估图留在 H100 RAM。

## 运行检查

正式启动读取模型配置和已登记资产，要求四张 H100。输入、loss、梯度和更新后的参数执行有限值检查；保存完整优化器、EMA、随机状态、样本位置和事件时间。资源限制和异常恢复继续执行项目规定。

临时迁移、冒烟测试和最大 batch 搜索产物在完成检查后删除；正式启动不依赖这些文件。迁移来源保存在 `docs/initializations.json` 及正式初始化资产的元数据中。新训练阶段与完整恢复状态记录在 `docs/training-history.json`。
