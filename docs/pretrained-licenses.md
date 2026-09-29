# SAFA FaceGen v1.0-pretrained：分项许可记录

核对日期：2026-09-29。对象是本仓库源码、`vendor/` 固定版本、六份选中 EMA 所沿用的初始化、运行所需 codec，以及 FFHQ/CelebA-HQ 数据来源。项目内事实依据为 [`initializations.json`](initializations.json)、[`artifacts.json`](artifacts.json)、`vendor/*/UPSTREAM.json`、各 vendor 许可文件及封版包 `THIRD_PARTY_NOTICES_zh.md`。本记录将**源码许可、第三方权重授权、训练数据条款**分别登记；模型训练所得 EMA 的公开分发条款仍需资产所有者按其来源确认。

| 分项 | 已核实的来源及条款 | 公开包处理 |
| --- | --- | --- |
| 项目自有源码 | 仓库根目录当前无项目级 `LICENSE`，自有部分的对外授权尚未指定。 | 由项目所有者确定源码许可与版权声明；保留第三方文件原声明。 |
| MeanFlow vendor | [Gsunshine/meanflow 固定提交](https://github.com/Gsunshine/meanflow/tree/d70cb55d298ee03c53bf6da67bec281082e4e2d9)；`vendor/meanflow/LICENSE` 为 Zhengyang Geng 的 MIT。 | 附 MIT 许可、版权声明和本地修改记录 `UPSTREAM.json`。三份 MeanFlow EMA 是项目训练资产，MIT 源码许可单独登记。 |
| LDM 与 Taming vendor | [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) 源码 MIT；`vendor/latent_diffusion/LICENSE` 载 Machine Vision and Learning Group, LMU Munich；`TAMING_LICENSE` 载 Patrick Esser、Robin Rombach、Björn Ommer 的 MIT 声明。 | 附两份原许可与 `UPSTREAM.json` 中的适配记录。 |
| LCM vendor | [luosiallen/latent-consistency-model](https://github.com/luosiallen/latent-consistency-model)；`vendor/latent_consistency/LICENSE` 为 Simian Luo 的 MIT。LCM 教师来自本项目 Diffusion 741 轮 EMA。 | 附 MIT 许可、署名和 `UPSTREAM.json`；教师与 LCM EMA 单列为项目权重。 |
| RF vendor 与求解器 | [gnobitab/RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) 仓库未提供统一 `LICENSE`；本地 `LICENSE_NOTICE.md` 记录保留文件中的 Google Research Apache-2.0 文件头，以及 NVlabs StyleGAN2 resampling 来源。Torch RK45 改写参照 SciPy，许可保存在 `vendor/rectified_flow/SCIPY_LICENSE.txt`。 | 逐文件保留原声明及修改记录。仓库缺少统一授权的文件需取得来源授权依据后确定对外源码包范围。 |
| RF NCSN++ 初始化 | [`initializations.json`](initializations.json) 指向 [官方 RF 仓库](https://github.com/gnobitab/RectifiedFlow) 与其 [checkpoint 下载项](https://drive.google.com/file/d/1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF/view)。官方 README 列有 CelebA-HQ 预训练 checkpoint；当前页面未列该 checkpoint 的独立再分发条款。 | 保存实际文件身份、来源链接与原说明；RF 项目 EMA 的权重发布条款待来源方/所有者核实。 |
| FFHQ LDM 初始化与 VQ-f4 codec | [`initializations.json`](initializations.json) 指向 CompVis 官方 [`ffhq.zip`](https://ommer-lab.com/files/latent-diffusion/ffhq.zip)；[官方模型表](https://github.com/CompVis/latent-diffusion#model-zoo)另列 [`vq-f4.zip`](https://ommer-lab.com/files/latent-diffusion/vq-f4.zip)。仓库源码 MIT；模型表未单列两份 checkpoint 的再分发许可。项目 Diffusion/LCM 使用 FFHQ LDM-VQ4 codec，实际文件身份需与封版资产登记对应。 | 将 LDM 初始化、VQ-f4 codec 及项目 Diffusion/LCM EMA 分别记录；核实实际 codec 文件所附条款与权利人答复后填写公开权重许可。 |
| SD-VAE-EMA codec | [Stability AI 官方模型卡](https://huggingface.co/stabilityai/sd-vae-ft-ema) 标注 `mit`；MeanFlow 使用该 codec。 | 对照实际下载 revision/文件，在公开包附 MIT 许可、来源和 Stability AI 署名。 |
| FFHQ 70,000 张 | [NVlabs 官方说明](https://github.com/NVlabs/ffhq-dataset#licenses)：单图分别为 CC BY 2.0、CC BY-NC 2.0、公有领域标记、CC0 或美国政府作品，单图作者与条款载于元数据；数据集集合、JSON、脚本和文档为 NVIDIA 的 CC BY-NC-SA 4.0，要求署名、标记改动及相同方式共享衍生数据集。官方还声明该数据集不用于开发或改进人脸识别技术。 | 数据保持私有；对实际引用图像保留逐图署名/条款元数据及 NVIDIA 来源。 |
| CelebA-HQ 30,000 张 | [CUHK CelebA 协议](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html#agreement) 限非商业研究、限制复制/发布/分发数据及其衍生数据；[CelebA-HQ 原始构建项目](https://github.com/tkarras/progressive_growing_of_gans) 将其材料标为 CC BY-NC 4.0，要求论文署名。 | 训练图与缓存保持私有；记录 CUHK 与 Karras 等作者来源。公开视频或样图及训练所得权重的具体用法由所有者按来源条款审定。 |

源码所附 MIT/Apache 文件头覆盖对应代码；数据和初始化 checkpoint 有各自的来源与条款。公开模型卡应给出每一权重及 codec 的来源、实际版本、许可依据、署名和使用范围，缺项见 [`pretrained-publication-draft.md`](pretrained-publication-draft.md)。
