# 原生接触边界头：受限训练与完整视频验证

## 状态与范围

新增入口 `scripts/run_contact_release.py` 提供互相隔离的 `train`、`render` 两阶段。它是针对当前固定源/驱动的**反事实边界变暗实验**，不是原生牙列 GT、通用牙齿分割或已通过的牙形优化。原模型权重不训练；默认 v1 推理不启用此头。

本地仅运行标准库、NumPy、AST 测试；不得在本地导入 Torch/CV2、部署环境、下载权重或运行模型。真实训练、581 帧重放、GPU/编码性能及画质尚需父代理按远程规范部署验证。本文不是服务器运行授权，不更改既有 GPU/空间限制。

**只有同一高清照片、同一原始驱动的完整 A0 / mild / strong 视频和同步对比，才能进入用户最终验收。训练指标、静帧、代码测试和 checkpoint 都不能替代成片；无收益不采用。**

## CLI 与输入隔离

两阶段共同参数：

- `--workspace`：已获准的项目工作区根；代码、输入、缓存及输出都必须位于其内。
- `--snapshot`：已验证的 production-FP16、581 帧 snapshot-r2。
- `--samples`：`prepare_contact_samples.py` 导出的 36 帧样本目录。
- `--output`：不存在的新目录，必须是 `--budget-root` 的严格子目录；不可覆盖旧产物。
- `--budget-root`：本轮共享预算目录；允许已有样本和前阶段产物，不允许与源码树/snapshot 重叠。
- `--authorize-contact`：显式启用此次受限实验。

`--stage train` 必须提供 `--labels`，不接受 `--checkpoint`。固定从新初始化头开始，不能热启动挑选结果。

`--stage render` 必须提供 `--checkpoint`，**提供任何 `--labels` 都在路径访问和张量导入前拒绝**。render 只读取 samples 的 `report.json` 作来源核验，不打开 sample NPZ、标签、目标 RGB、GT mask 或逐帧 correction。checkpoint 的 JSON 仅包含来源哈希和训练/验证帧号，不把帧号作为预测输入。

`prepare_contact_samples.load_inputs` 复核 snapshot 的完整性、581 K、F/xs、源媒体/五权重哈希及原 W/G 实现。render 的网络输入只有固定 F/xs/逐帧 K，当前和真实前帧 H 提取的 32 通道原生特征，以及训练后的共享 head。f0 的 previous=current。snapshot 的源 canvas/crop 矩阵仅用于原版 pasteback；不读 A0 成品 RGB 来预测或补牙。A0 MP4 只用于原片引用、音视频验证及三列比较编码。

## 运行守卫

复用 snapshot 的 `code_check`、`OwnedProcessGroup`、`budget`、音视频检查等，不另实现一套模型流水线。

- 必须 Linux、干净 Git 树、`PROBE_CODE_SHA=当前完整40位SHA`；训练与渲染代码 SHA 必须相同，便于审计/回退。
- `CUDA_VISIBLE_DEVICES` 必须是一个完整 GPU UUID。入口不选择 GPU、不检查他人资料；父代理运行前重新确认空闲 UUID 和用户总额度。
- `HOME/TMPDIR/TMP/TEMP/XDG_CACHE_HOME/TORCH_HOME/HF_HOME/CUDA_CACHE_PATH` 必须已隔离且存在于 workspace，`PYTHONDONTWRITEBYTECODE=1`。
- CPU affinity 至多 4 核；BLAS/OMP/编码器线程限制 4；原 production-FP16 W/G，确定性启用，禁 TF32，固定 snapshot 的 Torch/CUDA/NumPy 版本及随机种子。
- 每阶段 supervisor **600 秒**，本轮 budget-root 所有 `supervisor.json` 墙钟累计 **1800 秒**，失败也计入。不足完整 600 秒额度不启动。worker 在 585 秒处截止，留 15 秒做进程组清理和审计；实际超限仍记录失败，绝不截短/隐去计费时间。
- 保留 `release.lock` advisory lock 防止本入口并发抢占共同预算；运行时 supervisor 先记录 running。遗留 running/无效 supervisor 禁止继续，须父代理核实真实终止时间，不静默归零。
- 按现有 `du` 实际分配空间口径：累计新增 outputs ≤1 GiB，project ≤20 GiB，保留原 64 MiB 安全余量；监控期间超限停止自有进程组。父代理不能在同一预算根并发启动不遵循此锁的其它脚本。
- 不删旧产物，不全局安装，不网络下载，不终止他人进程。

失败报告与 supervisor 留在 output，`status=completed` 仅表示一次受限运行及审计完成，**不等于 `quality_pass`，更不等于视频采用**。

## labels 明确契约

`record.json` ≤1 MiB，字段：

```json
{
  "schema": "contact-boundary-counterfactual-v1",
  "purpose": "limited-source-boundary-experiment",
  "user_authorized_experiment": true,
  "human_semantic_mask_approved": false,
  "independent_review": "passed",
  "ROI": [200, 330, 320, 380],
  "samples_code_sha": "<samples report.current_code>",
  "samples_report_sha256": "<samples/report.json 文件SHA256>",
  "frames": [
    {
      "frame": 226,
      "split": "train",
      "file": "labels_f0226.npz",
      "sha256": "<标签NPZ文件SHA256>",
      "raw_sha256": "<samples report.files.raw512_f0226.png 文件SHA256>"
    }
  ]
}
```

实际 frames 必须覆盖 prepare 定义的全部 36 帧、各一次。`raw_sha256` **明确采用 PNG 文件 SHA256，不是带 shape/dtype 前缀的 `array_hash(raw)`**；对应 raw512 array hash 已在 samples/snapshot provenance 中另行核验。

每个 NPZ 压缩文件及展开内容都 ≤1 MiB，先 SHA、ZIP/NPY 声明大小检查，再安全加载；文件必须是 labels 内对应帧的确切 basename，`allow_pickle=False`。

四键均为 `[1,1,50,120]`：

- `allowed / protected / uncertain`：bool；`allowed & (protected | uncertain)` 必须为空。
- `target_delta`：float32、有限、范围 `[-0.6,0]`，非 allowed 严格为 0。

train 必含 225/226/227/228/251/252；validation 必含 254–259 连续段及 446/456，两 split 不交。225 为上下牙接触负例，251/252/292 为闭口负例，四者 allowed 与 delta 必须全零，不能把 225 描述为闭口。

保护与审美范围只准用于这个有限诊断实验；`human_semantic_mask_approved` 保持 false，不能升级成正式真实标签批准。

## r2：局部上下文门控对照

首版0f98499使用两层1×1，真实训练出现train负例误激活。r2只将backbone第二层改为3×3/padding1，加入局部边界上下文；末端gate/delta仍1×1，修正仍在所有计算之后乘gate、点对点加入原logits，因此predicted gate外数值支持域不扩大。shape变化另用arch=contact-boundary-context3-v2，重新初始化，不读取旧checkpoint。数据、固定阈值/强度、600+300步及留出不变。该修订依据train误激活，已有validation已被查看，因此不能将r2的相同留出称为完全未经开发观察的新测试集。

## 固定训练，不看验证调参

1. 只用 train features 计算 FP64 累积、FP32 保存的 32 通道 RMS，固定 `[1,32,1,1]` scale；模型自身保持 `1e-4` 下限。验证样本不进入 RMS。
2. A 阶段：FP32 head；Adam lr=.004，固定 600 步，只优化 backbone/gate_head。全 ROI 的 weighted BCE，正类权重 `min(50, negative/positive)`，protected/uncertain 同其它非 allowed 一样作为负类。delta 参数必须保持精确零。
3. 每步固定 seeded 4 帧 batch：轮转一个有 allowed 的 train 帧，加 seeded train 顺序的其它三项（允许抽到重复项）；frameID 只索引样本，不喂网络。每 100 步只输出 train loss、固定 .9 阈值的 precision/recall、保护域 gate 激活数。
4. A 后若任一正训练帧没有预测 gate 覆盖 allowed，记录 coverage_failure，不假装继续有效 delta 训练。
5. B：`freeze_gate()`，Adam lr=.01，仅 delta_head，固定 300 步。loss 为 `mean((correction-target)^2 on allowed) + .1*mean(correction^2 on nonallowed)`。这里 correction **来自预测 gate**，不是 GT gate teacher forcing。finite、非零梯度、实际参数更新及 backbone/gate 冻结逐步检查；数值/覆盖失败停止并明示。
6. 全部训练结束后才评估全部 36 帧，validation 不参与优化、阈值选择、早停或强度选择。报告分 split、不把有限同源验证当身份级留出。

保留 `candidate_checkpoint.pt`，只存 head.state_dict，无原模型权重。旁边 `.json` 保存 architecture、固定阈值/幅度、代码 blob/SHA、train/val IDs、snapshot 数组和原权重哈希、样本与标签来源哈希、失败列表。`weights_only=True` + strict 复载全部 36 样本 correction/gate/probability 及 FP16 ROI 输出要求完全一致。

train 输出每帧 `prediction_fNNNN.npz` 和 `predictedROI_fNNNN.png`。原 ROI logits 转回 FP16，correction 转 FP16 后相加 sigmoid；不制造由外部 raw 容器拼出的假模型全脸。不做输出后 RGB 复制。

报告分开记录：

- predicted gate 外 float/uint8 必须零差（数值支持域保证）；
- labels allowed 外、protected、uncertain 的 float/uint8 改变像素与 max（标签限定安全检查）；
- .9 阈值的 gate 覆盖与保护域误激活。

训练 `quality_pass` 的保守门槛要求全部有 allowed 的样本完整 gate 覆盖、保护/不确定域无 gate 激活、标签域外浮点/uint8 零变化，且无数值/覆盖失败。它仍不代表牙形观感通过。失败 checkpoint 可供失败可视化；不能称为优化模型。

## 真实 581 帧 render

从固定 F/xs/581 K 重放原 W/G，冻结参数并 eval；不用检测器、不重算源 F、不读取样本特征。每帧得到 H、真实前帧 H、原生 logits：

- baseline sigmoid 的 raw512、完整 pasteback RGB 哈希逐帧与 snapshot 一致，否则立刻终止；
- f0/225/266/446/580 检查 strength=0 和新 zero-head 与 baseline tensor 精确一致；
- mild=.5、strong=1 调用 `head.apply_to_logits`，保持原 FP16 base logits；不把 head 改成 RGB 后处理。

全 581 帧流式检查 predicted gate 外及固定 ROI 外 float/uint8 零差；完整画幅用 `warpAffine(pred_gate>0)>0` 的双线性传播域检查 uint8 域外零差。未做事后 RGB 强制替回。**这些是数值传播保证，不是完整全片上牙语义保证**，后者明确 unknown、待独立审阅。

产物：

- `mild_silent.mp4`、`strong_silent.mp4`：真实顺序 581 帧，25fps，CRF18/yuv420p。
- `mild_full.mp4`、`strong_full.mp4`：原 driver 音轨 streamcopy，不用 `-shortest`。
- `three_columns_full.mp4`：左 A0、中 mild、右 strong，各 panel 等比缩放/padding 至 512×512，原音轨 streamcopy。不烧录字幕，父页面必须注明列名和失败/待审状态。
- A0 直接引用 snapshot 的 `A0_full.mp4`，不另复制占预算，但上述本次 raw/full baseline 哈希已全帧对齐。
- `full_gate.npz`：bool `[581,1,50,120]`（展开约3.3 MiB），供独立审核，不存全量 H。
- prepare 的 36 个选定时刻分别保存 A0/mild/strong raw512 PNG（总108张）；完整画幅只保存226/255/446三时刻三版本。没有稀疏帧拼接冒充完整视频。

A0及五个新视频都经 ffprobe 和 ffmpeg 完整解码，要求 581 帧/25fps/23.24 秒；所有有声版本的音轨 packet SHA256、PTS/DTS/duration、流元数据与原 driver 相同。多路视频只做连续原帧缩放/拼列，不重复短片或变速。

render 报告永远 `quality_pass=false`：训练门槛通过时仍待完整视频视觉验收；训练已失败则明确 failure visualization only。同时记录每帧支持域、改动像素、原始/完整画幅哈希、W/G 参数与 buffers 前后哈希、五权重/媒体/代码前后哈希、checkpoint 来源、音频证据、显存峰值、CUDA 时间线秒数（包含 CPU 等待，不冒称纯 kernel 时间）、supervisor 墙钟和预算。父代理不能只拿 PNG 或数值宣告采用。

## 本地检查与父代理下一步

本地安全命令（不导入 Torch/CV2）：

```text
python -B -m unittest tests.test_contact_run tests.test_contact_prepare -v
```

需预设 `PYTHONDONTWRITEBYTECODE=1`。张量测试类仅 Linux 且显式 `CONTACT_RUN_TENSOR_TEST=1` 才导入 Torch；默认跳过。这是远程合成机制检查，不是标签训练或客户效果证明。

父代理顺序：审核三文件及 labels 契约 → 建立并发布 Git 回退点 → 按服务器规范部署同 SHA → 重新核资源和共享预算 → 验证样本/labels 来源 → train → 如失败仅以明确失败名义 render 取证 → 完整解码/音轨/语义审阅 → 完整三版本观看页及用户验收。不得因代码已写好跳过这些门槛。
