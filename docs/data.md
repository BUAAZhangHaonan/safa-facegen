# 数据、潜变量缓存与固定评价

H100 的训练集为 FFHQ 70,000 张和 CelebA-HQ 30,000 张，共 100,000 张 256×256 RGB JPEG95 图片。`data/hq256/manifest.json` 保存图片顺序、来源和文件元数据；`records[].path` 相对 `image_root`。原图保存在服务器本地，数据使用遵循 FFHQ 与 CelebA-HQ 各自的来源条款；模型初始化来源见 [initializations.json](initializations.json)，上游源码许可见各 `vendor` 目录。

2026-09-22 的全量源数据核验解码了 100,000 张图片：尺寸异常、解码失败和常量 RGB 图片均为 0；唯一图片为 99,973 张，27 条重复记录保留在固定训练集中。源报告位于 H100 的 `reports/data/hq256-20260922/`。图片目录现为 `data/hq256/images`。

## 确定性数据接口

`HQDataset(manifest, random_flip=True, seed=42, image_root=None)` 返回 CHW、float32、RGB `[-1,1]`。每轮调用 `set_epoch(epoch)`。水平翻转取 `blake2b(f"{seed}:{epoch}:{index}", digest_size=1)` 的最低位；样本索引和轮数固定时，DataLoader worker 数不会改变翻转结果。

`CachedLatentDataset` 使用同一翻转规则和只读 mmap。两类缓存的行序与数据 manifest 一致；每张图分别缓存原图编码结果和先翻转 RGB 后的编码结果。

| 模型家族 | 单条缓存形状与类型 | 内容 |
| --- | --- | --- |
| MeanFlow | `[2,8,32,32]` float32 | SD-VAE posterior mean/std；训练时重新采样并乘 0.18215 |
| Diffusion、Latent Consistency | `[2,3,64,64]` float32 | LDM-VQ4 连续 prequant latent；解码时执行 VQ 量化 |

缓存目录由 codec 与数据 manifest 身份区分。原初始化权重、两种 codec、两类潜变量缓存和完整 100k 数据均列入 [资产清单](artifacts.json)。

## 正式评价

每个 EMA 固定生成 1,024 张原图，按索引取前 256 张进行人工结构标注与联系表展示。第 `i` 张从 CPU float32 Torch RNG 的 `seed+i` 生成初始噪声与逐步噪声；评价 batch 改变时图像输入保持一致。参考集用同一 manifest 和 `NumPy RandomState(seed)` 无放回抽取 1,024 条。

FID1024、KID 使用 torch-fidelity 的 `inception-v3-compat` 2048 维特征；KID 使用 100 个大小为 1,000 的子集。单脸率采用 InsightFace `det_10g.onnx`，原生 256×256 输入、阈值 0.5，分母为 1,024。评价保留全部生成样本及逐图记录，另记录空白、多脸和非有限输出。质量比较使用相同参考图顺序、采样设置、特征提取器和权重。

`safa_facegen.evaluate` 写出原图、前 256 张联系表、生成与人脸检测逐条记录、参考记录、Inception 特征以及 `summary.json`。评价材料保存在服务器的 `reports/evaluation/`；本轮最终候选和选择见 [结果与选择](results.md) 与 [画质证据](quality.md)。
