# H100、K100 资产与评价流

H100 的项目根目录为 `/home/apulis-dev/code/meanflow_e15_h100_bundle`，承担四卡训练和完整状态保存。K100 的项目根目录为 `/home/k100/projects/safa-facegen`，保存副本、执行生成评价并向独立 SAFA 研究项目提供选中 EMA。四项适配共 55,000 次更新及最终副本同步已经完成；H100 当前没有训练作业，K100 空闲 worker 已于 2026-09-29 06:00 UTC 停止。

六模型选中 EMA、各模型最新可恢复完整状态共 11 个 checkpoint 身份，以及原始初始化、codec、100k 图像、两类潜变量缓存、正式原图和人工标注列在 [artifacts.json](artifacts.json)。中间权重整理以该清单为边界；评价结果和选择见 [results.md](results.md)。

## 保存与传输

检查点名称由模型、已完成 HQ 轮数和 UTC 时间组成；元数据保存精确更新数、样本曝光、数据与 codec 身份、来源和阶段配方。MeanFlow 的完整状态为带 `manifest.json` 和 `COMPLETE.json` 的目录，PyTorch 模型保存完整状态、独立 EMA 和配套配置。导出的 EMA 由生成器严格加载后用于正式评价。

训练器在 `runs/<model_id>/requests.jsonl` 记录保存和审核请求。K100 的复制 worker 按请求身份建立临时副本，完成内容核对后发布，并向 H100 写入传输回执。`runs/controller/quality/` 保存本轮阶段登记与恢复记录；每个模型的 `latest-<stage_id>.json` 指向对应阶段的完整状态。历史阶段与目标 ID 保留在原始 checkpoint、登记和事件记录中。

`reports/evaluation/<checkpoint_id>/review/` 保存 1,024 张正式评价原图、指标及原始记录；`quality/` 子目录保存纹理评分、覆盖率和人工审核材料。画质协议见 [quality.md](quality.md)，固定参考数据见 [data.md](data.md)。

## 本机配置与调用

机器路径和复制配置位于 Git 忽略的 `configs/local.json`、`configs/replication.local.json`。按模型登记的 `quality_evaluation` 指向 Diffusion Min-SNR 的固定评价协议。复制进程经 stdin 接收跳板和 H100 凭据；普通配置只保存路径与评价参数。正式训练控制器要求显式 `--campaign`，对应 campaign 中仅登记一个模型。

项目中的 `requirements/h100-jax.lock`、`h100-torch.lock` 和 `k100-torch.lock` 记录两台机器的实际依赖版本。源码与运行材料分别管理；服务器上的保留清单提供可定位的 checkpoint、图像和标注。
