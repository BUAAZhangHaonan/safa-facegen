# SAFA FaceGen v1.0-pretrained 公开发布草稿

状态：**内部封版后的公开页草稿，尚未指定平台与可见性，也未登记全部权重的公开分发授权。** 本文只据现有设置准备页面内容；具体资产身份和分项依据见 [`pretrained-licenses.md`](pretrained-licenses.md)、[`initializations.json`](initializations.json) 及封版包 `RELEASE_README_zh.md`/`THIRD_PARTY_NOTICES_zh.md`。公开动作由项目所有者确定。

## 建议页面正文

**SAFA FaceGen v1.0-pretrained** 提供六种无条件 256×256 RGB 人脸生成器。默认入口为 Diffusion-LDM-UNet（DDIM 200 步，eta=1）；四步入口为 LatentConsistency-LDM-UNet。完整目录通过 `release.json` 指定模型 EMA、codec 与采样参数；`release-smoke.json` 记录内部加载、生成和噪声梯度验收。PyTorch 接口冻结参数，支持输入噪声的梯度传播。六份 EMA 的历史目标、轮次及画质局限以随包模型卡和封版 `RELEASE_README_zh.md` 为准。

训练数据来自 FFHQ 70,000 张与 CelebA-HQ 30,000 张。训练图、缓存、优化器状态及完整恢复档案由项目私有保存。FFHQ 的单图许可和集合许可分别适用；CelebA 来源限定非商业研究并限制数据再分发。使用者应阅读[分项许可记录](pretrained-licenses.md)与原始来源。生成图像可能存在纹理覆盖、软化、肤色偏差和局部五官问题；模型卡记录对应候选的既有评价与观察。

## 公开页面待填字段

| 字段 | 当前可填写内容 / 所有者决定项 |
| --- | --- |
| 平台、仓库/模型页 URL、账户 | 尚未指定；由所有者选择。 |
| 可见性与访问条件 | 尚未指定；由所有者决定公开范围、申请或下载方式。 |
| 包版本与文件身份 | `v1.0-pretrained`；依据 `release.json` 填六份 EMA、两类 codec、既有来源身份及文件字节数，ZIP 文件与条目数见 `release-assets.json`。 |
| 项目自有源码许可 | 尚未指定；由所有者填写项目级许可及版权声明。 |
| 第三方源码署名 | 保留 MeanFlow、CompVis LDM/Taming、LCM、RF 各文件原署名及许可/来源说明；RF 无统一仓库许可，逐文件确定分发范围。 |
| 初始化/衍生权重许可 | RF NCSN++、FFHQ LDM、VQ-f4 官方下载页未列独立再分发条款；逐项取得可引用的授权依据，并确定六份项目 EMA 的公开权重许可与使用范围。 |
| SD-VAE-EMA | 官方模型卡标 MIT；按实际下载版本附来源、许可与署名。 |
| 数据声明 | 填 FFHQ 单图/集合条款、CelebA 与 CelebA-HQ 来源；训练图和缓存不列入公开包。 |
| 模型卡 | 每模型列权重身份、初始化链、训练数据、采样参数、质量证据与局限；沿用封版记录的结果，不新增未执行评价。 |

发布包中的许可文件须与实际包含的源码、权重和 codec 一一对应。官方依据：[FFHQ](https://github.com/NVlabs/ffhq-dataset#licenses)、[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html#agreement)、[CelebA-HQ 构建项目](https://github.com/tkarras/progressive_growing_of_gans)、[RF](https://github.com/gnobitab/RectifiedFlow)、[CompVis LDM 模型表](https://github.com/CompVis/latent-diffusion#model-zoo)、[SD-VAE-EMA 模型卡](https://huggingface.co/stabilityai/sd-vae-ft-ema)。
