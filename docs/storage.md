# H100 到 K100 的同步与评价

H100 负责预训练，工程根目录固定为 `/home/apulis-dev/code/meanflow_e15_h100_bundle`；K100 工程根目录固定为 `/home/k100/projects/safa-facegen`。`safa_facegen.replicate` 在 K100 运行，通过 aliyun 跳板连接 H100，只读取 `runs/`、`models/` 与 `data/hq256/`。H100 写入位置限定为 `reports/replication/receipts/` 的传输回执与 `reports/replication/active-transfer.json` 的保护记录；它不修改研究工程，也不操作旧资产目录。

## 完整性与身份

新模型身份统一为 `<模型名称>-<已完成HQ轮数>ep-<UTC时间>`，例如 `MeanFlow-B-4-0003ep-20260922T150000Z`。MeanFlow 的时间可含微秒。阶段完成轮数以训练器实际处理的 HQ 样本数除以 manifest 条数向下取整；原始训练 lineage 保存在 checkpoint 内，不用旧实验编号命名新版本。

训练器在完成原子保存后向 `runs/<model_id>/requests.jsonl` 写入 `save`、`preview` 或 `review`。H100 本地完整保存约每 900 秒，预览导出请求约每 1,800 秒，质量审核请求约每 7,200 秒触发。

K100 本地配置使用 `poll_seconds: 300`、`restore_transfer_interval_seconds: 7200` 和 `transfer_previews: false`，减少跨服务器传输。每个模型距离上一次完整恢复副本传输成功至少 2 小时，才开始下一次完整状态传输；首份恢复副本可以立即传输。等待期间选择最新的未开始保存请求。审核 EMA 仍按每 2 小时的请求传输并生成 1,024 张评价图片；单独的预览权重请求记为 `skipped_by_policy`。H100 继续保存检查点和生成导出请求。

已经开始或暂停的传输保留部分文件并继续完成，不受新任务的间隔限制。模型结束时的完整状态也遵循该传输间隔，H100 保留源状态直到 K100 确认接收；训练切换无需等待这次传输。恢复副本间隔按模型分别计算，审核请求保持优先级。

轮询 requests 文件时遇到 SSH 断连、超时或 socket 关闭，会立即结束当前连接，在 `status.json` 记录 `connection_retry`，并按默认30秒轮询周期重新连接；这些故障没有请求内容，不生成 rejected/failed 模型请求。收到的内容存在 JSON、路径、身份或同 ID 内容冲突时，仍保留明确失败记录。网络恢复不会自动改写历史失败记录；旧版本误登记的 transport 拒绝项须先按具体 ID、错误与 payload 核实，再由主控制流程单独归档，不能批量清除真正的模型失败。

传输耗时可能大于保存间隔。每完成或暂停一个产物后，worker 刷新请求并按 `review EMA → preview EMA → 完整恢复` 排序。同类请求优先继续已开始的传输；尚未开始的 save/preview 只保留该模型最新一份，旧请求明确标为 superseded，之后不再读取已淘汰的源状态。所有 review 请求保留，包括每两小时产生的不同审核候选。

完整恢复和 preview 文件复制期间，每约30秒在1 MiB分块边界检查新请求。出现更高优先级 EMA 时，先 flush/fsync 当前 partial，登记 `paused_transfer` 并释放文件句柄，再复制 EMA。新 review 因而不必等数 GiB 的完整状态传完；实际响应时间还受当前网络读取及请求元数据往返限制。正在复制的 review 不会被后来的 preview/save 打断。这里保证调度优先级，不能保证低带宽下每个候选都在两小时间隔内完成传输。

受控暂停保留当前进程内的 SHA256 累积状态，续传先核对源身份和本地大小/mtime/ctime，再从已有字节偏移继续，不反复扫描前缀。已经完整核对的分片在 partial 的 REPLICA 元数据中登记 stat，恢复时直接复用。worker 重启或网络重连丢失内存状态时，仅对尚未完成文件的已有前缀重新计算一次，再继续后续字节；最终仍必须匹配源登记的完整 SHA。SFTP 每次最多预读1 MiB、32个并发请求，读完该窗口后才允许暂停，避免关闭文件后旧预取线程继续占用连接。网络失败按30至900秒退避。partial 从不作为完整模型或可评价权重发布。

MeanFlow checkpoint 是包含 `state/`、`manifest.json` 与 `COMPLETE.json` 的目录。worker 验证 COMPLETE 内的 metadata 身份，并复用 checkpoint 的 `state_files` 清单中已登记的分片 SHA256，流式写入 K100 时核对传输内容；不在每轮同步前重新扫描 H100 大文件。EMA 同样复用导出 manifest 中的 `ema.safetensors` 身份。普通 `save` 可以不含 EMA 导出。

Torch checkpoint 是同一身份的 `.state.pt`、`.ema.pt` 和 `.config.json`。request 必须有 `complete:true` 与 `hashes:{state,ema,config}`，且三个文件均须位于对应模型的 run 目录。所有请求均拒绝路径穿越、越界、符号链接与 partial/tmp 产物。传输在本地唯一 partial 目录完成，全部哈希一致后才原子发布。

SHA256 与完整标记证明传输内容一致，不能替代恢复训练的实跑测试。因此同步完成状态叫 `transport_verified`，记录 `restore_exercised:false`。EMA 评价必须实际严格加载模型并生成图像；任何加载或指标失败都留下 failed 状态，不发布新的有效审核指针。

本地文件验证后，worker 向 H100 `reports/replication/receipts/<request_id>.json` 原子写入回执，再读回核验。回执包括 `request_id`、`model_id`、`identity`、`event`、`artifact_role`、`source_checkpoint`、`source_ema`、`source_declared_hashes`、`files[{source,path,bytes,mtime,sha256}]`、`received_complete`、`checkpoint_received` 与时间。同 ID 回执不可改写为不同内容；失败时本地已传输文件保留。

源控制器把某份完整恢复视为已转存时必须同时确认：`event="save"`、`artifact_role="restore"`、`received_complete=true`、`checkpoint_received=true`，模型身份与源 checkpoint 路径完全匹配，文件哈希与源内容一致。`preview/review` 的 EMA 回执不能作为完整恢复状态已转存的证据。模型至少有一份有效完整恢复回执后，才可按下述 latest/active/retry 保护策略清理旧恢复状态，包括带宽限制下未开始复制而被 superseded 的状态。所有仍在等待用户决定的 EMA 与正式 approved EMA 必须保留；MeanFlow 的旧恢复状态可清理 `state/`，但仍有待审核导出时必须保留 `COMPLETE.json`、`manifest.json` 和 `export/`，供后续严格校验。`event="probe"` 回执用于传输链路验证，不得用于清理。

`active-transfer.json` 的 `status="active"` 表示正在复制，携带 `request_id`、`identity`、`model_id`、`source_checkpoint`、`event` 与 `updated_at_unix`，每约30秒更新。结束后 `status="idle"` 且清空当前对象。`retrying` 数组包含网络重试和调度暂停两类待续传身份，分别登记 `status="retry_transfer"` 或 `"paused_transfer"`。暂停时先持久化队列状态，再将 active 转为 retrying 保护；EMA 传输期间该保护持续存在，恢复时再转回 active。已开始的暂停/重试状态不会被新 save 静默 superseded。源控制器必须保护 latest、active 与 retrying 中的完整状态；不能仅因一次心跳延迟删除活动源文件。已有至少一次完整且验证通过的恢复回执后，才允许按主控制流程策略清理其余已被明确 superseded 的旧恢复状态，EMA 审核边界保持不变。

## 本地保存与审核边界

`models/replicas/<model_id>/restore/<identity>/` 保存完整恢复副本；只有新副本完整通过哈希验证并推进 latest_restore 指针后，worker 才移除自己创建的上一个 restore 目录。删除前核对路径边界和 `REPLICA.json` 管理标记。首次使用前已有资产、初始化权重、codec、EMA 审核目录和研究文件均不属于这项轮换。

`models/replicas/<model_id>/ema/<identity>/` 保存待评价和已发布候选。尚未开始的旧 preview 可以标记为 superseded；review 不自动合并。已发布审核候选及其评价结果保留，直到用户明确 decision 或授权替换后的清理。新 review 完成真实生成和全部指标后才更新 `current_review`；旧审核对象不会因为新权重抵达而被删除。审核积压会暂存多份 EMA，需要在用户作出决定后由主控制流程收束为正式 EMA 与最新完整恢复副本。

`reports/replication/journal.sqlite3` 用 WAL 记录每个 request 的身份、状态、传输位置、失败原因与审核指针。同 ID、同内容重复到达不会重复启动；同 ID 指向不同内容直接失败。进程中断留下的 copying 状态进入可续传队列，paused_transfer 保持可续传，evaluating 标为 failed 等待显式处理。哈希不一致、模型加载失败、磁盘不足或指标失败均不会被当作网络问题重试，也不自动降低参数。`reports/replication/status.json` 提供最后轮询时间、待评价、失败、网络重试与调度暂停数量。

`latest_restore` 按同模型 checkpoint 身份中的 UTC 时间（相同时按完整身份）单调更新。网络退避造成旧 save 晚于新 save 完成时，保留旧请求的传输回执并清理其已完成本地副本，不回退恢复指针，也不删除较新的完整恢复文件。

CPU 复制循环和单 GPU 评价线程分开运行。K100 GPU0 每次只运行一个正式任务，优先处理 review，避免预览占满审核队列。`reports/evaluation/<identity>/preview/` 存固定 64 张预览；`review/` 存固定 1,024 张及 FID1024、KID、单脸率，详情见 [data.md](data.md)。生成器是真实新权重；没有可加载的权重时，任务明确失败。

## 启动配置与凭据

先安装 `paramiko>=3.5` 以及评价环境。普通配置只放本地非敏感路径和参数，例如：

```json
{
  "evaluate": true,
  "poll_seconds": 30,
  "device": "cuda:0",
  "batch_size": 8,
  "cpu_threads": 8,
  "dataset_manifest": "data/hq256/manifest.json",
  "image_root": "data/hq256/images",
  "inception_weights": "models/evaluation/weights-inception-2015-12-05-6726825d.pth",
  "face_detector": "models/evaluation/det_10g.onnx",
  "models": {
    "MeanFlow-B-4": {"codec": "models/codecs/SD-VAE-EMA"},
    "MeanFlow-B-2": {"codec": "models/codecs/SD-VAE-EMA"},
    "MeanFlow-L-2": {"codec": "models/codecs/SD-VAE-EMA"},
    "RectifiedFlow-NCSNpp": {"codec": null},
    "Diffusion-LDM-UNet": {"codec": "models/codecs/LDM-VQ4.pt"},
    "LatentConsistency-LDM-UNet": {"codec": "models/codecs/LDM-VQ4.pt"}
  }
}
```

codec、参考图片和评价权重必须已在 K100 项目中存在；这些路径应来自部署后的依赖注册表，不从 checkpoint 内的 H100 绝对路径推断。大模型所需的评价 batch 必须显式配置，失败不会自动减小。

启动器以管道向 `python -m safa_facegen.replicate --config <配置路径>` 的 stdin 发送一行 JSON：`{jump:{hostname,port,user,password,key},h100:{hostname,port,user,password,key}}`。`key` 是已核验 SSH 公钥完整二进制 blob 的 base64。两级主机都执行严格 pin 校验。密码不写入配置、环境变量、命令行、日志或临时文件；启动器只在内存构造并注入。worker 不承担凭据持久化或自动重启注入工作。

`--once` 仅用于 `evaluate:false` 的单次同步检查。本地 `.tools/start_replication.py` 从已有 SSH 配置与 known_hosts 在内存读取凭据，通过 SSH stdin 调用 K100 `.tools/replication_launch.py`。不加 `--once` 时，后者以匿名 stdin 管道将凭据交给独立 session 的 worker；加 `--once` 时只执行同步检查，不启动 GPU 评价。普通配置为 `configs/replication.local.json`，其中不包含密码。长期任务由主控制流程启动；worker 用本地文件锁拒绝第二个进程抢占同一队列。失败详情和请求状态保留，恢复或重试需主控制流程明确处理。

源端收到更新完整恢复的有效回执后，可同时删除旧状态对应的 save-only EMA；preview/review 候选及已审批的正式 EMA 继续保留。训练 metadata 保留，并用 `recovery_state_available`、`ema_available` 明确登记文件是否仍可用。

## K100 研究材料保留边界

2026-09-22 对照 `/home/k100/projects/safa-research` 当前实现后，仅迁移它缺少的 E0 编码器、非测试特征缓存、对应索引、被索引引用的 materialized 图片及所需 codec。旧生成器训练实现、旧测试和完整源码快照不迁移；同名实现冲突保持研究项目当前版本。详细保留条目由研究项目 `docs/retained-expression-materials.md` 说明，运行材料位于其忽略的 `artifacts/`、`data/`。

四组特征有明确224像素预处理配置，按原张量转换为当前 metadata schema；另一组143081条全脸特征未找到完整缓存预处理来源，保留但不启用。当前研究配置的表达资产和 codec 路径按迁移结果修复；生成器初始化和评价 checkpoint 尚待明确选择，不能把这些配置声明为可直接训练。

研究依赖的 `/home/hdd3/zhanghaonan/AffectNet`、`/home/k100/Datasets/Face`、`/home/k100/.insightface` 必须保留。已替代 facegen 文件的删除由主控制流程执行；replicate worker 不删除旧研究资产。

复制时复用源 checkpoint 已登记的 SHA，流式写入时验证一次目标内容；无变化的已验证副本复用文件 stat，不重新扫描整份权重。MeanFlow restore 还要求 state 文件集合及大小与提交清单完全相符，拒绝已被源端回收的 STATE_RETIRED 标记。SFTP 使用最多32个并发预取请求，内存随预取窗口受限。
