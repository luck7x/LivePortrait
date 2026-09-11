# 上牙真实能力实验：源快照与完整 A0

## 当前范围与状态

本阶段只实现 `scripts/snapshot_upper_teeth.py`，不实现 E1、适配器训练或新牙形生成。A0 是原 v1 生产路径基线，不是优化视频。代码基于已拆出 `G.forward_features/decode_features` 的工作树；实际部署必须以干净 Git 和 `PROBE_CODE_SHA` 固定，不接受漂移源码。

本地只允许 stdlib、NumPy、AST 检查；本文件不表示已经在 CUDA 上执行或生成 A0。远程部署、资源授权、原高清闭嘴照片和原始驱动的选择由父代理负责。最终牙形优化仍必须交付同输入完整 581 帧候选、v1 和同步对比，不能以本快照、短片或指标替代。

## 执行契约

CLI 必需参数：

```text
python -B scripts/snapshot_upper_teeth.py \
  --workspace "$WORKSPACE" \
  --source "$SOURCE" --driving "$DRIVING" \
  --budget-root "$BUDGET_ROOT" --output "$OUTPUT" \
  --authorize-experimental
```

这里只使用占位环境变量，不内嵌服务器地址或私人绝对路径。

- 只允许 Linux，必须显式授权，`CUDA_VISIBLE_DEVICES` 必须是单个 `GPU-...` UUID。运行前父代理另查空闲 UUID 和用户总 GPU 额度。
- 代码、输入、模型和所有写入路径须在授权 workspace 内；解析符号链接后复核。budget-root 与代码目录分离，output 必须是其下新的子目录，输入不得放入 budget-root。不覆盖、不删除失败产物。
- 必须预置 `HOME/TMPDIR/TMP/TEMP/XDG_CACHE_HOME/TORCH_HOME/HF_HOME/CUDA_CACHE_PATH` 到 workspace 内已有隔离目录，`PYTHONDONTWRITEBYTECODE=1`。不安装、下载模型或建立环境。五个原权重、landmark.onnx 和完整 buffalo_l 五个 ONNX 必须已存在。
- 原模型加载仍走原 Pipeline；临时包装 `Module.load_state_dict`，即便调用方选择宽松加载，也必须检查 missing/unexpected keys 均为空。不忽略加载差异。
- 自有 worker 独立进程组，600秒硬超时清理自有组（含ffmpeg）。使用waitid/WNOWAIT检测组长退出，先清理自有组再回收组长PID，避免遗漏后代或向复用后的PID发信号；finally恢复全部monkeypatch。仅自有 worker 设置至多 4 个可用 CPU 的 affinity，Torch/CV2、BLAS 和自有 ffmpeg 线程设为 4，不修改共享进程或全局环境。
- GNU `du -s -B1` 使用真实分配空间，单次遍历自动去重硬链接。workspace 总上限 20GiB，budget-root **累计**上限 1GiB，不是每个 output 各 1GiB；运行保留 64MiB 安全余量，并为累计剩余额度检查 workspace 与磁盘空间。父代理监督循环约每 0.5 秒复查，worker 每25帧也检查。不能把这类轮询说成文件系统硬配额；突然外部写入仍可能导致中止。
- `report.json.wall_seconds` 在父进程收尾时更新为总墙钟；失败也留下 `supervisor.json` 的时间及退出码。父代理须把失败耗时也计入 E0/E1 合计 ≤1800 秒；此脚本没有实施 E1，也不替父代理维护跨任务时间账本。

## 数据通路：只执行一次原 Pipeline

显式设置相对运动、stitch、normalize_lip 为 True，multiplier=1；eye/lip retarget、source-video eye retarget、driving crop 为 False；expression-friendly、all、FP16、torchcompile=False。CropConfig 按原 CLI ArgumentConfig 的同名字段解析，例如 det_thresh=0.15。

`cfg.flag_pasteback=False` 仅避免 Pipeline 累积 581 份高清全幅内存。真正输出仍对每帧原 `wrapper.warp_decode` 返回值调用原 `parse_output`，随后使用原 `prepare_paste_back/paste_back` 和真实 source canvas 输出全画幅；未复制模型公式，也没有自行实现浮点转 uint8。

- 原 `cropper.crop_source_image` 只调用一次：记录其收到的真实 source_canvas（原 Pipeline 已 resize/even 裁边），以及完整数值 crop_info，包括变换矩阵、原 crop、256 crop 和 landmarks。
- 原 `extract_feature_3d` 记录实际 source_input 与 F，仅一次；每帧检查 F 和 x_s 固定，保存最终 finalK，而不是重新计算 motion/stitching。
- 原 `make_motion_template` 记录 581 个实际 M 输入数组 hash，以及 motion、eye/lip ratio 数值。不保存 581 份 M 图片。
- `G.forward_features` 只在 f225、228、251、252、264、266、292、446、456 保存 H。要求真实原生 H 是 FP16 `[1,64,256,256]`，不把 FP32 先转换后冒充生产数据。
- W/G 始终一起走原 `wrapper.warp_decode` 内部 inference_ctx。没有复用旧bringup的W32+G16路线，也未另行重写autocast。显式固定seed=20260911、deterministic=True、cudnn.benchmark=False、禁用TF32及CUBLAS_WORKSPACE_CONFIG=:4096:8；这是固定数值环境下调用原Pipeline的本次A0，后续臂必须继承。不能宣称它与历史v1成片逐像素相同。
- 在本次 execute 生命周期内抑制 Pipeline 的 `dump` 写原 driving 旁 pkl，记录被拦截模板；`images2video/concat_frames` 不产生重复输出，`has_audio_stream=False` 仅禁用它的尾部音频操作。Pipeline 的原 512 输出列表仍可留内存。没有用空白帧生成交付视频。

## 文件与核验

| 文件 | 内容 |
|---|---|
| `source_snapshot.npz` | source_input `[1,3,256,256]`、F `[1,32,16,64,64]`、x_s `[1,21,3]`、final_k `[581,1,21,3]`；均为真实 float32 返回值 |
| `source_crop.npz` | 完整数值 crop_info；NPZ 不包含 object/pickle |
| `source_canvas.png` | 真实 resize/even 后源全幅 RGB |
| `driving_motion.npz` | 原 make_motion_template 的 motion 和 ratio 数值 |
| `H_fNNNN.npz` | 仅指定 9 帧 FP16 H，总量约 72MiB |
| `raw512_fNNNN.png` | 相同 9 帧原 parse_output 的原始 RGB |
| `full_f0000/0266/0446.png` | 3 张真实 pasteback 全幅预览 |
| `A0_full_silent.mp4` | 每帧 RGB 直接流式输入 ffmpeg，CRF18/yuv420p/25fps |
| `A0_full.mp4` | silent 视频流 + 原驱动第一条音轨，全部 stream copy，不使用 shortest |
| `report.json` | 配置、来源、逐数组/逐帧 hash、模型参数/buffers、文件 hash、输入和五权重首尾 hash、版本、GPU UUID/显存、预算/时间与媒体核验 |
| `supervisor.json` | 成败退出码及总墙钟；失败产物保留，不可当作完成快照 |

原驱动先验证 1024×1024、581 帧、25fps、23.24 秒且有音轨；输出两视频均核验 581 帧、25fps、23.24 秒并完整 ffmpeg decode。音轨比较所有 packet payload SHA256、PTS、DTS、duration 和音频 stream 时间基/时长等元数据，必须完全一致，否则任务失败，不悄悄转码或移动时间戳。

所有快照数组 hash 包含 shape、dtype、连续字节；report 同时含输出文件 SHA256（report 不对自身做递归 hash）。原模型参数和全部 buffers 必须首尾一致，五权重/两输入首尾 hash 必须相同。

## 已有本地检查与远程门禁

本地检查包含15项测试（新增不回收组长PID前清理子进程组、晚到定时器不误发信号测试）；未导入 Torch/CV2，未执行模型。安全命令：

```text
python -B -m unittest discover -s tests -p test_upper_snapshot.py -v
```

测试覆盖参数授权默认、非 Linux 早拒绝、路径约束、预算累计与余量、shape/dtype/非有限数、数组 hash、视频/音频契约以及 AST 的原 Pipeline 单次执行、hook、进程组超时和禁止重写 autocast。测试不 import Torch、CV2 或 src 模型。

远程接受报告必须同时满足 `status=completed`、`supervisor_verified=true` 和 `supervisor.json.returncode=0`，不能仅看 worker 先写入的完成字段。

仍需父代理在获准远程环境执行：实际 Pipeline/FP16 H 契约、600 秒内完成、完整原音轨 packet 比较、真实 du、完整 A0 全幅成片核验。若原始音轨时间戳无法被容器原样保留，脚本严格失败，应分析原始/输出时间线而不是放宽通过条件。A0 只是后续同源实验的基准，不能当作客户牙形改善证据。
