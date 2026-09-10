# 牙齿模型本体因果诊断探针

## 性质与固定范围

`scripts/probe_teeth_causality.py` 是**模型本体扰动诊断脚本**，不是优化器、像素后处理、训练或已验证修复。它不修改 `src/`，不覆盖 v1 权重或原始媒体，不将旧 trace 或外部 pickle 喂给 pipeline。

固定用户原驱动的 **f0 + f260–267**，25fps，共 9 个内部输入；用户可查看的输出仅为 **f260–267 连续 8 帧（10.40–10.68 秒，视频时长 0.32 秒）**。四种 case 共 32 个目标模型输出，不是完整 581 帧／23.24 秒视频优化。

| case | 唯一预定扰动 |
|---|---|
| baseline | 原始 M 输入与运动模板 |
| freeze_lip | 除 f0 外，仅把 `exp[:, [6,12,14,17,19,20], :]` 固定为本次 baseline f264 的值 |
| freeze_pose | 除 f0 外，仅把 `R / t / scale` 固定为本次 baseline f264 的值；exp 原样 |
| blur_driver | 仅在原 `make_motion_template` 接到的真实 `I_d` 上，对 f260–267 做 256→128（AREA）、5×5 Gaussian（sigma=0）、128→256（LINEAR）；f0 与源照片不变 |

冻结模板的 `kp/x_s` 等未干预字段保持原值，**不会为追求模板内部的几何自洽而额外改写它们**。最终关键点公式、relative motion、运动 multiplier、stitching、lip-normalize、W/G 均由原 `LivePortraitPipeline.execute` 执行，不在脚本中重抄或替换。

## 参数与复用机制

- photo、relative、stitching、lip-normalize 为 true；采用原 CLI photo 的 `expression-friendly`、multiplier=1、animation_region=all、FP16 默认；source crop det_thresh=0.15，其余原 crop 默认。完整配置和每 case arguments 均记录。
- `crop_driving_video=False`；要求驱动是 1024×1024、25fps，否则拒绝，不暗中改用裁剪驱动。
- source-video-eye / eye-retarget / lip-retarget 为 false。lip-normalize 自身所需的原 retarget_lip 路径仍保留。
- pasteback 关闭，只保存 W/G 的原生 512×512 RGB。模块级 `images2video` 在探针作用域临时禁写并 finally 恢复，因此不生成包含 f0 时间跳跃的伪成片；不改变运动或模型输出。
- baseline 真正调用 `make_motion_template`，在内存保存 deepcopy；两种 freeze 只复制此缓存；blur 重新真实运行 M。所有 case 仍以内部 AVI 进入原 execute，不从外部 pkl 加载。
- F 第一次真实提取后缓存。以后只有 `prepare_source` 产物 tensor 的 shape/dtype/bytes hash 完全相同才复用；所有 warp 调用核对 `f_s/x_s` hash，四 case 不同立即失败。
- 固定 seed=20260910，Torch 导入前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。pipeline 导入会打开 cudnn benchmark，脚本在导入后明确关回，并打开 deterministic algorithms（不使用 warn_only）。若现有 CUDA 算子不支持确定性，则失败保留证据，不静默降级。
- baseline f263 除原始 warp 外，再用完全相同的 F、x_s、最终 K 调用两次原 warp，记录 uint8 精确相等及 MAE，并保存两张 repeat PNG。文件字节不同不是像素不同；报告比较的是解码数组。

## 输入与产物

通过原项目src.utils.io.load_video（ImageIO/FFmpeg）顺序读取原视频到f267，选择f0与f260–267九帧进入模型；此前各帧只参与解码不参与本探针推理。检查25fps和原始1024 RGB尺寸。不能改用OpenCV seek读取RGB后就宣称是同一生产解码路径：其色度上采样可产生差异。原帧未经缩放，以FFmpeg FFV1/BGR0写入内部AVI，再由同一load_video读取，要求九帧RGB完全相等且无多余帧。此等值是视频解码RGB，不声称恢复压缩前相机数据。

```text
output/
  probe.json                         # 状态、args、hash、invariants、proxy、预算
  internal/
    NOT_FOR_DELIVERY.txt
    rawsubset_ANCHOR_TIME_JUMP.avi    # f0→f260 时间跳跃，不可作为交付视频
    source_feature.npz               # 本次真实 source tensor、F 数值
    <case>/
      inputs/rawsubset_ANCHOR_TIME_JUMP.avi  # 本次内部硬链接
      inputs/rawsubset_ANCHOR_TIME_JUMP.pkl  # 原 pipeline 本次自行写出；不回读
      motion_numeric.npz             # 实际 M 输入、模板所有 motion 字段、眼/唇比率
      f000000.png                    # 内部 anchor 输出
      fXXXXXX_finalK.npz             # 每次最终 K 和 source K
      f263_repeat1.png               # 仅 baseline
      f263_repeat2.png
      suppressed_video/              # 原 execute 可能创建的空目录，无成片
  baseline/                          # 以下每个 case 均相同输出布局
  freeze_lip/
  freeze_pose/
  blur_driver/
    f000260.png ... f000267.png       # 真正连续 pre-pasteback/pre-video raw512 RGB
    diagnostic_f260-f267_8frames_silent.mp4
```

每 case 单独 inputs 路径，避免原 pipeline 的 `.pkl` 同名覆盖。数值 npz 不含 Python 对象；外部分析读取时使用 `allow_pickle=False`。运动各字段 baseline/干预后 hash、两种 freeze 的全部非目标字段不变断言、anchor 不变断言都记录；完整前后数值可由 baseline 和 case npz 对照。正常完成还验证四 case anchor 输出像素一致。

诊断 MP4 使用 8 张真实连续 PNG 编码为 H.264/yuv420p，无音轨，并完整解码核对 8 帧；有损编码不保证逐像素等于 PNG。每 case 对 baseline 的全图 RGB MAE、口区 `[190,310,360,430]`（xyxy，右/下边界不含）MAE，及 7 对连续相邻帧的相同指标仅为**响应差异 proxy，不是牙齿质量分数**。不把冻结导致的少运动说成稳定修复，不依据 MAE 排名自然度。

## 执行门槛（本文不构成服务器授权）

本地只允许静态与 NumPy 测试。脚本拒绝非 Linux 或缺少 `--authorize-probe` 的运行。未来执行前仍须按工作区服务器规范重新核验空闲 GPU UUID、用户总 GPU 额度、原视频/照片身份与已批准的运行阶段；脚本不能替代共享服务器调度与人工授权。

远程预先满足：

1. 已批准的项目 workspace；源图、驱动、代码、五份 F/M/W/G/S 权重、实际使用的裁剪模型路径以及隔离 HOME/TMPDIR/HF_HOME/TORCH_HOME/XDG_CACHE_HOME 全部 resolve 在 workspace 内。insightface 子路径也检查 symlink 逃逸。
2. 部署 commit 已由父代理审核发布；设置 `PROBE_CODE_SHA` 为该**完整 40 位 SHA**。运行前后 HEAD 必须精确匹配，并且 `git status --porcelain --untracked-files=all` 为空。脚本不做 commit、pull、push 或 checkout。
3. output 必须不存在，并且在 workspace 内、干净代码 worktree 外。不会自动清理失败目录，不会覆盖已有输出。
4. `CUDA_VISIBLE_DEVICES` 仅一个已批准设备；实际 Torch 必须只见一张 CUDA 卡。这里只验证可见数，不声称其他用户任务空闲。
5. 复用既有 Torch、CV2、Pillow、NumPy、FFmpeg/libx264/FFV1、GNU du 和原 pipeline 依赖，不安装或下载。
6. `du -s -B1` 按实际分配块计量，同一遍历不重复累计硬链接；不用逻辑文件长度，也不使用 `--apparent-size`。项目最多 20GiB、探针输出最多 512MiB，预先保留剩余输出额度的项目/磁盘 headroom；每次主要写入前检查预留量，完成再审计。内部 AVI 用硬链接避免四份实际存储。
7. SIGALRM 与所有直接 subprocess 的剩余时间 timeout 限制 300 秒；CLI 另以独立进程组 supervisor 在约 299 秒终止仍未完成的工作，随后最多两次各 0.5 秒 TERM/KILL 等待。该硬保护避免阻塞的原生 CUDA 调用拖过上限，只终止本次自建子进程组，不停止已有进程。极端内核不可中断 I/O 不属于 Python 可保证的实时截止。

在以上条件已获授权后，命令形式为（仅示意，**本文未执行**）：

```text
python -B scripts/probe_teeth_causality.py --workspace <项目绝对路径> --source <原照片绝对路径> --driving <原驱动绝对路径> --output <新诊断目录绝对路径> --authorize-probe
```

运行前后记录源照片/原驱动 SHA256、code SHA 与 Git clean、五份权重 SHA256并要求不变。任何异常记 `failed`、保留已有 PNG/数值/JSON，不删除、不将失败当结果。超时或其他中断可能无法做完整结束 hash 审计；`postflight_verified=false` 明确表示不具备成功验收证据。进程刚开始或非正常 OS 终止可能只留下最初 `running` 记录，应按未完成处理。

## 本地验证及剩余验收

允许的本地命令：

```text
python -B -m unittest discover -s tests -p test_teeth_causality.py -v
```

本次本地执行结果：**15 项测试全部通过**（真实运行；不含远程依赖）。测试只导入标准库与 NumPy：冻结 lip/pose 的字段边界、f264 索引、f0 与原缓存不变、proxy 算术/ROI、哈希、路径边界、平台/授权拒绝、mock du/headroom/Git pin，以及 AST/源码级 hook 和延迟 import 检查。

本地测试**不能证明**远程模型构造、确定性 CUDA、FFV1 RGB 往返、真实 M 输入模糊、四 case F/x_s 一致性、重复 warp 像素、MP4 完整解码、真实时空资源预算已经通过；这些检查已写入脚本，但必须在未来获准的固定代码服务器运行中取证。也未运行训练、模型推理、CV2、Torch、服务器或网络操作。因果响应即使可复现，也不独自证明牙齿异常的唯一根因或任何可采用优化。
