# 真实视频监督：原生 G 尾段复制训练与完整视频验证

## 当前交付边界

新增 `scripts/run_real_teeth.py`、`tests/test_real_teeth_run.py` 和本文。未修改原 G、`RealTeethDecoder`、数据准备或标签工具；未提交、推送、联网、部署或执行模型。

脚本的三个阶段已实现，不是 render 占位：

- `train`：复制原 G 的 `up_1` 与 `conv_img` 参数，先训练独立语义 mask，再用未修改的真实 GT RGB 训练复制出的学生层。
- `render-validation`：验证视频自身固定 F/x_s/581 个绝对 K，独立重放 581 帧；GT/BASE/轻/强四列。
- `render-cross`：固定 `8465cf4a619dd8381c7be1f421de31c4e6ecd27c` 高清快照的源 F/x_s/K 与原 pasteback；A0/轻/强三列。

这不是旧 scalar 暗线头，不读取 A 视频 RGB 作为模型输入，不逐帧优化 delta，不把 GT 填入候选。原 F/M/W/G/S 权重文件保持不变，只实例化并冻结原 W/G；不重新执行 F/M、人脸检测、源图关键点提取。真实 GT 是各视频自身固定 frame0 仿射对齐的 RGB，**不是跨人物牙列的原生真值**。

代码完成不等于运行验证或牙齿优化完成。`status=completed` 仅表示该阶段执行和数值/媒体检查完成；`quality_pass` 始终保留 false，完整视频还必须由父代理和用户审阅。

## 输入合同与隔离

公共必填参数：

```text
--stage train|render-validation|render-cross
--workspace <批准的项目根>
--data <prepare_real_teeth 输出目录>
--budget-root <本阶段统一预算根，已包含 data/labels 及其 supervisor>
--output <budget-root 内全新目录>
--authorize-real
```

阶段专用参数：

| 阶段 | 必须 | 拒绝 |
|---|---|---|
| train | `--labels <标签目录>` | checkpoint、cross-snapshot |
| render-validation | `--checkpoint <decoder.pt>` | labels、cross-snapshot |
| render-cross | `--checkpoint <decoder.pt>`、`--cross-snapshot <8465快照目录>` | labels |

标签目录须有 `record.json`，符合 `real-teeth-supervision-v1` / `real-video-paired-supervision`；raw GT unchanged、实验授权及独立审阅 passed 必须为真，human semantic labels approved 仍为 false。每个 split 的全部 136 个 selected 帧均需 `allowed/protected/unknown` 三个 bool `[512,512]` 数组，互斥、完备，allowed 只能位于固定 ROI `[160,290,350,410]`。标签通过 data report SHA256 与 data code SHA 绑定。没有正例相邻训练帧，立即以 data_gate 失败，不生成替代 teacher。

真实数据的 `files` 是 `path -> {bytes,sha256}`，不是历史快照的字符串 hash 表。`videos.train/validation.source_arrays` 绑定各自 `source.npz` 四个源数组；cropM/M_o2c/M_c2o/pt_crop106 另验，不传给旧四 key snapshot validator。581 个 K 逐个验证。

训练前核验全部数据文件 hash、源媒体/原权重和原 G/util/W/dense-motion/crop 等代码 blob。渲染读取 data report，并验证源 NPZ、源文件、原权重及 encoded 比较视频；**不打开 GT/BASE 单帧 PNG、标注 NPZ 或样本 feature NPZ**，这些项只保留已绑定 report 的文件大小检查。渲染可以读取编码后的真实 GT 视频作为 ffmpeg 对照，不能进入 forward。训练 checkpoint 的 JSON 内保存标签来源记录，不在渲染中访问标签目录。

## 固定训练方案

1. 随机种子、Torch/CUDA/NumPy 版本、deterministic/TF32 配置与 data 环境一致；原参数 FP32，原 W/G 与 student CUDA autocast FP16，mask/scale FP32，禁止整个 G `.half()`。
2. 流式遍历 **train 的 136 份**原生 `[1,16,120,190]` FP16 ROI，逐通道 sum-of-squares/count，计算 RMS，最小 `1e-4`。不堆叠两段视频，不读取验证 feature 计算归一化。
3. 原 G 尾段复制在任何更新前完成。在 train f0/225/228/251/255/266/446 重放 W/G，对学生初始 logits、sigmoid tensor、uint8 检查精确一致。这里是远程生产 GPU FP16检查；本地不执行 Torch，不能冒称已经通过 CPU/GPU 实测。
4. Mask 固定 600 步 Adam，lr=.003，batch4，每 batch 至少一个 allowed 非空训练帧。weighted BCE 仅 known=allowed|protected，unknown 不作负例；正权重 `min(20, neg/pos)`，均只统计 train ROI。
5. 冻结 mask，切 `set_training_stage('student')`。仅 `student_up_1`、`student_conv` 固定 300 步 Adam，lr=1e-5。相邻对来自 train 连续区，至少一帧 allowed 非空；每对两个 batch1 forward，避免批量大小改变 production AMP 基准。
6. 原始 student sigmoid 对未改 GT 计算：allowed L1 + .2×allowed 邻接像素梯度 L1 + .5×protected 对原 base 蒸馏 L1 + .1×共同 allowed 区内相邻预测差与相邻 GT 差的 L1。unknown 没有 GT 监督；时序项不是简单压低运动。GT 只进 loss，不进 student/mask forward。训练损失作用于 raw student，**不是用 GT gate 合成后再训练**。
7. 两阶段启用 GradScaler，检查每步 loss/gradient 有限且梯度非零；结束检查 mask 确更新、学生两个模块均改变、学生阶段 mask 不变。记录 step100/300 等学习曲线，不保存额外中间权重，不按验证集挑 checkpoint。
8. 仅保存最终 `delta_state_dict()` 和 provenance JSON。之后才评估 train/validation 全部缓存 gate；两 split 各七个固定时刻重放真实 W/G/student，分别报告 base/raw-student/gated 与真实 GT 的分区 MAE、保护区/unknown 实际变更、预测 gate 误激活与数值域外差。几帧 train+validation 通过 `weights_only=True` 严格增量复载，tensor/uint8 都必须精确一致。

`RealTeethDecoder.mask_features(H)` 对 H 的激活在 FP32 执行，而准备阶段缓存是 FP16 激活后保存；因此脚本同时记录固定重放点的真实 gate 与缓存 gate 统计，**不声称两者逐位一致**。若门槛附近出现语义误激活，应作为实际模型门槛失败处理，不能调验证阈值掩盖。该细节来自既有模块 API，本任务未修改模块。

任何 protected/unknown 预测激活、全零覆盖，均不能判语义通过。数值 gate 外零差不证明 gate 本身是牙齿。

## 独立完整视频阶段

两个 render 都重新从相同 checkpoint 加载同一个共享学生，依次处理 0..580：

- 输入只来自各自固定源 F/x_s/K 经原 W，再 `extract_base -> student_logits / mask_features / predict_mask -> compose`。
- validation 每帧 base raw hash 必须等于 data validation 记录；cross 每帧 base raw 和 pasteback full hash 均必须等于 HD snapshot。
- 每帧检查 strength0 精确回原 base；轻档 .5、强档 1。每档均验证 predicted gate 外及固定 ROI 外 float/uint8 最大差 0，cross 另验证双线性 pasteback 传播域外 uint8 最大差 0。
- 两路流式 ffmpeg encoder，不缓存全视频 H，不输出整段 PNG 序列。保存 36 个选定时刻的三列 raw 拼图 PNG 和全片 bool gate `[581,1,120,190]`；拼图只是辅助，不充当视频。
- validation 输出完整 512 画布 `mild_full.mp4`、`strong_full.mp4`、`four_columns_full.mp4`。cross 两个独立候选是原高清画幅，同步三列每列缩放/补边至512观看。
- 复用不可变 BASE/A0 编码视频作为基线列，但本次仍逐帧重算并核验 base raw；不把复用视频当候选。
- 原源视频音轨 `-c:a copy` / stream copy，严格比较 packet hash、PTS、DTS、duration 与 stream 字段。无 `-shortest`。每个 MP4 必须 581 帧、25fps、23.24 秒并完整 ffmpeg decode，通过后才记录 completed。
- 预测不读 labels、GT PNG、样本特征，不调用原 M/F，不按帧号注入形态补丁。frame index 仅用于固定 K 顺序和取证。

## 资源与失败守卫

仅 Linux 且显式授权；单一 `CUDA_VISIBLE_DEVICES=GPU-UUID`，worker CPU affinity4，OMP/MKL/BLAS4线程，缓存/HOME/TMP 均须预先隔离在 workspace。`PYTHONDONTWRITEBYTECODE=1`、`-B`、CUBLAS deterministic 环境强制检查。`PROBE_CODE_SHA` 必须对应干净当前 Git HEAD，部署与固定 SHA 由父代理负责。

复用 `OwnedProcessGroup`、原空间 budget 和累计 supervisor charge：

- 每个 stage 最长600秒；同 budget-root 所有 completed/failed supervisor（包括 data、labels、失败尝试）累计不超过1800秒；未结束 supervisor 拒绝继续。
- **训练预留两个 render 各600秒**，实际训练上限为 `min(600, 600 - prior_charge)`，不暗中减少600/300训练步数。例如 data/labels 已用150秒，train 只剩450秒；若 prior≥600，训练直接拒绝。不能用更换预算根逃避累计账本。实际耗时未知，固定训练可能超时失败；须由父代理在部署前确认已有耗时和剩余额度。
- render 最长 `min(600, 1800-prior_charge)`；超时/超空间停止当前 owned group，保存 failure/supervisor，保留部分产物，不删除、不覆盖旧目录、不把失败编码当完成。
- 统一新增1GiB/项目20GiB，复用原64MiB余量；每阶段循环和父监督持续检查。父代理仍需每次查 GPU UUID/用户总额度，本脚本不调查或停止他人进程。
- report inventory 排除自身、supervisor、活动 worker.log、failure 和临时 JSON，避免自引用和日志变化导致伪 hash。不输出巨量调试帧。

## 本地检查与父代理待验

本地允许且已运行的检查：

```text
python -B -m unittest discover -s tests -p test_real_teeth_run.py -v
```

覆盖 CLI 互斥、累计预算/失败计费、标签完备互斥、真实相邻配对、train-only RMS/正例 batch、NumPy 真实 GT 时序损失与 unknown 隔离、render 输入隔离、固定步数/阈值、增量 strict load 及无 eager Torch/CV2 导入。它们不是 CUDA/ffmpeg 运行证据。

父代理部署后仍需：真实 data/labels 合同验收；原 FP16 step0/replay 哈希核验；固定训练在剩余额度内完成；AMP 梯度及增量复载；两阶段完整编码/音轨/空间/保护域核验；对完整候选进行语义及原速视觉审阅。尤其不能把真实同人物重建监督自动解释为高清跨人物牙齿 GT，也不能以固定 ROI/预测 gate 外零差取代嘴唇、下牙保护验收。没有完整视频或视频没有净收益，均不发布修复版。
