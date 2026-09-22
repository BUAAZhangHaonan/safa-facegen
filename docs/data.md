# HQ256 数据、潜变量缓存与固定评价

训练集保留 FFHQ 70,000 张与 CelebA-HQ 30,000 张，共 100,000 条来源记录。输入使用 H100 原项目已有的 256×256 RGB JPEG95 文件。此次核验不重新裁剪、缩放或按人脸检测结果筛选。重复图片登记在报告中，继续保留原来源记录和固定训练集大小。

H100 项目根目录仍是 `/home/apulis-dev/code/meanflow_e15_h100_bundle`。`data/hq256/manifest.json` 中的 `image_root` 相对项目根目录，`records[].path` 相对图片根目录。部署到 K100 或后续整理目录时，可用 `--image-root` 覆盖图片位置；文件内容和 manifest 哈希仍需校验。不要在训练或缓存构建期间移动源文件。

## 已完成的源数据核验

2026-09-22 已对 `data/face_256_q95/{ffhq,celeba_hq}` 全量解码，并计算文件与解码后 RGB 像素的 SHA256：

| 检查 | 结果 |
|---|---:|
| 完整解码与尺寸合格 | 100,000 / 100,000 |
| FFHQ / CelebA-HQ | 70,000 / 30,000 |
| 无法解码或尺寸异常 | 0 |
| 常量 RGB 图片 | 0 |
| 唯一文件 SHA256 / RGB SHA256 | 99,973 / 99,973 |
| 重复记录 | 27，全部保留 |

初次 manifest SHA256 为 `3807eea7fcd8270c323184f7a794213042210fd124e14e2df9e0e56597ac47e4`。随后图片目录迁移为 `data/hq256/images`，manifest 仅更新图片根路径后，当前有效 SHA256 为 `bbabc8bf00d5b92e75323ab2fc10c70453851892aa2d85caa0e78b80390bb3d0`，仍为 100,000 条记录。源证据位于 H100 `reports/data/hq256-20260922/summary.json`、`duplicates.jsonl`、`invalid-images.jsonl`。该检查证明这些既有 256 图片可读且完整；不重新证明上游下载许可或与原始高分辨率图片逐像素一致。来源许可记录由项目数据来源文档管理。

重新核验必须使用新的 manifest 文件名和报告目录，不覆盖原证据：

```bash
python -m safa_facegen.data \
  --project-root "$PWD" --image-root "$PWD/data/hq256/images" \
  --output "$PWD/data/hq256/manifest-rechecked.json" \
  --report-dir "$PWD/reports/data/hq256-rechecked" --workers 8
```

核验器逐张读取完整像素，拒绝 256×256 之外的尺寸，并检查文件在读取期间是否变化。任何失败均保留错误报告、返回非零状态，不发布可用于训练的新 manifest。

## 训练数据接口

`HQDataset(manifest, random_flip=True, seed=42, image_root=None)` 返回 CHW、float32、RGB `[-1,1]`。每轮调用 `set_epoch(epoch)`。水平翻转取 `blake2b(f"{seed}:{epoch}:{index}", digest_size=1)` 最低位；与 DataLoader 的 worker 数和进程 RNG 无关。epoch 放在共享进程状态中，persistent workers 也能看到更新。

`CachedLatentDataset(cache_manifest, random_flip=True, seed=42)` 使用同样的翻转规则。文件只读 mmap，每次只复制选中样本的一个视图，不把整份缓存装入内存。缓存行序与数据 manifest 的 records 顺序一致。

## 按 codec 身份分开的缓存

每个缓存目录由 codec 哈希和数据 manifest 哈希确定。缓存的编码始终执行两次：原图，以及**先水平翻转 RGB 图片再编码**。不以翻转已有潜变量代替第二次编码。

| 家族 | 每条记录缓存 | 数值语义 | 100k 数组大小 |
|---|---|---|---:|
| MeanFlow | `[2,8,32,32]` float32 | 前 4 通道 posterior mean，后 4 通道 posterior std；均未 scaled | 6.55 GB |
| LDM | `[2,3,64,64]` float32 | VQModelInterface 编码的连续 prequant latent；scale=1 | 9.83 GB |

MeanFlow 训练器每次取样后重新采样 `mean + std * epsilon`，按官方训练配方应用潜变量缩放。缓存中不提前固定 posterior 随机样本。LDM 缓存不执行 VQ quantization，解码时由 codec 执行官方量化步骤。

缓存 manifest 的核心字段为 `schema_version`、`status`、`dataset_manifest_sha256`、`dataset_manifest_stat`、`codec.{family,sha256,files}`、`array`、`array_bytes`、`shape`、`dtype`、`layout="N,F,C,H,W"`、`representation="mean_std"|"prequant"`。`array` 相对缓存 manifest 所在目录。只有 `status="complete"` 可用于训练；失败保留 partial 文件与状态，不自动重试或减小 batch。读取缓存只核对已登记身份、文件大小及 NumPy header，不扫描完整数组。

构建前先指定已登记的本地 codec 路径与哈希；禁止隐式下载：

```bash
python -m safa_facegen.cache \
  --manifest data/hq256/manifest.json --output-root data/hq256/cache \
  --family meanflow --codec-checkpoint models/codecs/SD-VAE-EMA \
  --expected-codec-sha256 "$MF_CODEC_SHA256" --device cuda:0 --batch-size 32

python -m safa_facegen.cache \
  --manifest data/hq256/manifest.json --output-root data/hq256/cache \
  --family ldm --codec-checkpoint models/codecs/LDM-VQ4.pt \
  --codec-factory safa_facegen.torch_models:load_codec \
  --expected-codec-sha256 "$LDM_CODEC_SHA256" --device cuda:0 --batch-size 32
```

示例路径必须替换为依赖注册表中的实际位置。codec 身份复用下载/迁移时登记的 sibling JSON：文件 `LDM-VQ4.pt` 对应 `LDM-VQ4.json`，目录 `SD-VAE-EMA` 对应 `SD-VAE-EMA.json`。运行时只核对大小与登记的文件 stat；不反复计算权重哈希。图片复用首次全量核验结果，缓存构建只检查 bytes/mtime，不逐图重哈希，也不在完成时扫描整份数组。

中断后，在原构建命令追加 `--resume-manifest <原缓存目录>/manifest.json`。仅接受 building/failed 状态，保持原路径及 codec/数据身份，从最后已 flush/fsync 并写入 manifest 的 `rows_completed` 继续；边界之后未提交的行会重写。每100个 batch 持久化进度，完成后原子改名为 `latents.npy` 并发布 complete。构建的峰值内存由编码 batch 决定；训练 mmap 的系统页面缓存仍须计入机器的内存预算。

## K100 固定评价

评价只接受导出的 EMA 权重。生成器需声明 `state_role="ema"` 与 `ema_sha256`，提供 `noise_shape`、`sample(noise, step_noises=None, grad_enabled=False)`，输出 `[N,3,256,256]` RGB `[-1,1]`。随机采样器还需声明 `step_noise_count` 和 `step_noise_shape` 并使用显式提供的逐步噪声。

每次固定生成 1,024 张，前 256 张按索引直接排列为联系表。第 i 张使用独立 CPU float32 Torch RNG，seed 为 `seed+i`；在同一流中先抽初始噪声，再按顺序抽逐步噪声。改变评价 batch 不改变这些输入噪声。逐步噪声超过 512 MiB 时明确失败，由操作者显式选较小评价 batch；不会自动调整训练或评价参数。

参考集使用同一数据 manifest，由 NumPy RandomState(seed) 无放回选出 1,024 条记录；记录其索引与原始哈希。所有有限输出都参加评价，包括空白、无脸和多脸图片。空白定义为 RGB 各通道空间标准差的最大值小于 1 个 uint8 级别，不用于过滤。非有限原始输出另存 `.npy`，联系表位置以红框标注；存在非有限输出时 FID/KID 明确失败，绝不对占位 PNG 报分。

K100 只需镜像 seed=42 选出的 1,024 张参考图片，保留其相对路径及逐文件哈希；manifest 仍使用 H100 的完整 100k 原文件，以维持同一身份哈希。该目录必须登记为评价子集镜像，不能宣称完整训练集。评价直接加载固定索引对应路径，因此不要求其余 98,976 张图片存在；若改用其它参考 seed，需先明确镜像对应图片，缺失时直接失败。

FID1024 与 KID 使用 torch-fidelity 的 `inception-v3-compat` 2048 维特征及其官方 FID/KID 实现；内部使用该实现的图像缩放，不替换为普通 torchvision Inception。KID 使用 100 个大小为 1,000 的子集。样本数较小，FID1024 存在明显有限样本偏差；只能在相同参考集、样本数、权重和协议下比较，不等价于 FID50k 或质量达标证明。

单脸率由本地 InsightFace `det_10g.onnx`、原生256×256 检测输入、阈值 0.5 得出，分母始终为 1,024；不向训练输入添加检测、身份、landmark 或其它人脸条件。ONNX 使用 CPUExecutionProvider，直接创建推理会话并限制为8个计算线程；官方 RetinaFace 接收该会话。模型与特征提取权重必须事先存在，缺失时失败，不联网补下载。

```bash
python -m safa_facegen.evaluate \
  --model-id MeanFlow-B-2 --checkpoint "$EMA_CHECKPOINT" \
  --codec "$CODEC_PATH" --dataset-manifest data/hq256/manifest.json \
  --image-root "$HQ_IMAGE_ROOT" --output "$NEW_EVALUATION_DIRECTORY" \
  --inception-weights models/evaluation/weights-inception-2015-12-05-6726825d.pth \
  --face-detector models/evaluation/det_10g.onnx \
  --device cuda:0 --batch-size 8 --seed 42 --cpu-threads 8
```

评价目录拒绝覆盖。输出包含完整 PNG、`first-0256-contact-sheet.png`、生成与人脸检测逐条记录、参考数据记录、生成和参考 Inception 特征 mmap、均值/协方差及 `summary.json`。summary 复用 checkpoint 导出/传输时登记的唯一 EMA 身份和数据/评价权重来源，记录软件版本、采样协议和失败信息。评价权重从 `models/evaluation/assets.json` 读取登记身份并核对 stat；不重新扫描 checkpoint、固定参考图片和模型权重，不为每张输出 PNG 计算 SHA。任一必需指标失败，进程返回非零，保留已产生的证据。

2026-09-22 在 K100 `.venv-torch`（Torch 2.11.0、torch-fidelity 0.4.0）已通过 CPU 合约检查：图像归一化、persistent workers 跨轮确定性翻转、两种 mmap 表示与哈希、跨 batch 相同初始及逐步噪声；已有官方 Inception 权重和 InsightFace ONNX 均完成真实前向调用。新训练模型的质量审核使用对应 EMA 生成的 1,024 张样本，并单独保存评价材料。

人脸检测使用生成图原生分辨率，避免将近景人脸放大到640×640后产生尺度相关漏检。检测阈值保持0.5，全部1024张图片参与计数；原始图片和FID/KID采样设置保持不变。纠正已有报告时记录原检测设置、原统计和纠正原因。
