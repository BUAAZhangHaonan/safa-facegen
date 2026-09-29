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

实际运行验收及 ZIP 大小将在完成后写入收尾回执。
