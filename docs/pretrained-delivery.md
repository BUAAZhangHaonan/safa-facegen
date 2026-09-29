# SAFA FaceGen v1.0-pretrained 交付记录

新增训练更新数为 0。六份锁定核心 EMA 的来源见 `configs/pretrained-release.json`，默认 Diffusion 741 轮，少步原 LCD 19,732 步。

## 资产与执行位置

- K100 装配目录：`/home/k100/releases/staging/SAFA_FaceGen_v1.0-pretrained`。
- 迁移验收目录：`/home/k100/releases/validated/SAFA_FaceGen_v1.0-pretrained`。
- 七份 ZIP 输出：`/home/k100/releases/release-assets`。
- 收尾工具：`/home/k100/closeout/SAFA_FaceGen_Closeout_20260929`。
- 现有正式运行环境：`/home/k100/projects/safa-facegen/.venv-torch/bin/python`。

装配目录使用 K100 `/dev/sda3` 的 ext4 文件系统。验收在整个目录迁移后，以新 Python 进程逐模型执行，每模型两个接口输出。精度固定 FP32、CUDA/cuDNN TF32 关闭、matmul highest。

## 恢复档案与算力

H100 恢复档案复用现有 NFS 项目卷：`/home/apulis-dev/code/meanflow_e15_h100_bundle`。85 个 checkpoint 文件、11 个身份共 38,113,077,366 字节，存在、字节数、JSON 与 PyTorch ZIP 容器检查通过。原始初始化、codec、100k 图像、两类 latent 缓存位于同一 NFS 项目卷。详情见 [持久化回执](pretrained-archive-receipt.json)。

K100 保留正式评价材料和 AI 标签：`/home/k100/projects/safa-facegen/reports/evaluation`，固定图册位于 `reports/final`；磁盘为 `/dev/sda3` ext4。

H100 计算实例可由用户停止/释放，平台操作保留项目 NFS PVC 及目录，卷删除策略选择 retain。用户提供的算力时价为 40 美元/小时，本次训练费用预算为 0。实例释放由用户操作，当前记录状态为待用户执行。

## 发布许可

本地预训练交付的来源、可加载性、生成和噪声梯度验收单独记录。历史质量门、AI 标签、uncertain 和 reviewer 保留原值。许可依据见 [分项许可](pretrained-licenses.md)；公开平台、可见性及部分权重分发授权见 [公开页草稿](pretrained-publication-draft.md)。

实际验收于 2026-09-29T09:34:20.497426+00:00 完成，六模型全部通过，12 张接口输出已保存。发布状态为 `PRETRAINED_READY`；完整记录见 [交付记录](pretrained-release.json) 和 [ZIP 索引](pretrained-assets.json)。

## 实际验收

源码及打包内容使用提交 `742758d1d58f465f5f9bd5edfbd3d1ce10320a51`，标签 `v1.0-pretrained`。本文件及收尾回执随后的提交用于登记产物。

| 模型 | 初始噪声梯度范数 | 有限逐步噪声梯度数量 | 结果 |
| --- | ---: | ---: | --- |
| MeanFlow-B-4 | 0.001994830789 | 0 | passed |
| MeanFlow-B-2 | 0.004089429043 | 0 | passed |
| MeanFlow-L-2 | 0.003040174721 | 0 | passed |
| Diffusion-LDM-UNet | 0.000114947361 | 200 | passed |
| RectifiedFlow-NCSNpp | 0.001324526034 | 0 | passed |
| LatentConsistency-LDM-UNet | 0.0008015792118 | 3 | passed |

输出均为 float32 `[1,3,256,256]`、RGB、有限、范围 [-1,1]。生成器参数冻结，参数梯度为空。验收一次通过；记录在 K100 发布目录的 `release-smoke.json`，本地副本为 `reports/pretrained/release-smoke.json`。

## 七份实际 ZIP

共同目录：`/home/k100/releases/release-assets`。总大小 5899236695 字节。

| 文件名 | 字节数 | ZIP 条目数 |
| --- | ---: | ---: |
| `SAFA_FaceGen_v1.0-pretrained__runtime_and_codecs.zip` | 558,299,779 | 137 |
| `SAFA_FaceGen_v1.0-pretrained__MeanFlow-B-4.zip` | 524,678,652 | 3 |
| `SAFA_FaceGen_v1.0-pretrained__MeanFlow-B-2.zip` | 524,383,546 | 3 |
| `SAFA_FaceGen_v1.0-pretrained__MeanFlow-L-2.zip` | 1,836,578,024 | 3 |
| `SAFA_FaceGen_v1.0-pretrained__Diffusion-LDM-UNet.zip` | 1,096,377,894 | 5 |
| `SAFA_FaceGen_v1.0-pretrained__RectifiedFlow-NCSNpp.zip` | 262,540,523 | 3 |
| `SAFA_FaceGen_v1.0-pretrained__LatentConsistency-LDM-UNet.zip` | 1,096,378,277 | 3 |

七份文件解压到同一父目录，即组成 `SAFA_FaceGen_v1.0-pretrained`。已核对每份 ZIP 的字节数、中央目录条目数及共同根目录。
