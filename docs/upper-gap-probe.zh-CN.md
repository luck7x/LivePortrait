# 上牙窄黑缝视频后处理实验（未验证视觉收益）

这是固定 v1 **最终完整画幅 MP4** 的 CPU 后处理，不是手工单帧设计、模型推理或训练。不会覆盖输入、权重或原有产物。仅处理真实连续窗口；不会补帧，也不宣称牙列结构、语义分割或自然度优化完成。

## 运行边界

实际处理只允许获准 Linux 环境；本地仅 NumPy 合成测试与静态检查。OpenCV 在 runner 通过授权参数和路径检查后延迟导入；不新增依赖、不安装 OpenCV/ffmpeg。运行服务器前仍须读取工作区服务器规范，并另行核实授权、资源和代码固定版本。

示例（路径仅为占位，应替换成已授权根内的真实路径）：

```text
python scripts/probe_upper_gap.py --workspace /authorized/project \
  --authorize-experimental-preview \
  --input /authorized/project/v1-final.mp4 \
  --output /authorized/project/results/new-gap-probe \
  --start 225 --end 275 --anchor 266 \
  --upper-roi X Y WIDTH HEIGHT --strength 0.85 --ema
```

- 帧号从 0 开始，start/end 均包含，anchor 必须在其中；窗口最多 96 帧，源视频最多 600 帧。为确认总长度，顺序读取源视频，但仅缓存选段 RGB，单帧宽高均不超过 1280。
- ROI 是 **anchor 完整画幅坐标** 的整排可见上牙包围框，不是旧 512 面板坐标；只给上牙，不包含下牙。输入未经人工审核的 ROI 不因此成为已批准遮罩。
- workspace 必须显式指定；输入与输出 resolve 后均在其中。输出路径不得存在，父目录必须已存在。工作区体积加保守剩余产物预算不得超过 20 GiB；每帧检查磁盘余量和剩余预算。处理及编码以 300 秒期限约束，ffmpeg 使用原环境程序、2 编码线程和 2 filter-complex 线程，不覆盖文件。
- 失败直接报错，保留已写出的部分证据，不自动删除或覆盖；没有最终 metrics.json 的目录不算成功。Python/CV 单个底层调用不能被 Python 时间检查硬中断；时间检查在解码、逐帧和写入/编码边界执行，子进程使用剩余期限 timeout。

## 机制与保守回退

1. 将全图缩小至最长边 640，以 ROI 上方眼眉/上脸区域提取角点（排除嘴部）。anchor→current LK 加反向误差过滤，RANSAC partial affine；至少 8 内点、比例至少 0.6，误差不超过 `1.5 × ROI宽/75` 原图像素，缩放限 0.75–1.25。旋转限20度、平移限100原画幅像素，特征限制在鼻眼和上脸附近，避免背景主导。任何失败回退原帧 A，清空 EMA。
2. 当前 RGB 对齐 anchor；只取 ROI 左右 `5×scale`、上 `4×scale`、下 `18×scale` 上下文。上下文越出有效映射区域则回退。
3. NumPy 根据低红度浅色像素找分开的亮带。上牙须接近 ROI 中心并在 ROI 内；下面必须存在第二亮带和至少约 `2×scale` 的明显暗隙。红色像素额外排除。单牙带、闭口或歧义不修。
4. 仅上牙各行首尾 seed 之间，以约 `5×scale`、最大 15 的奇数水平 closing 新增区域为候选 fill。只有新增 fill 内混合邻近 seed 加权 RGB，默认强度 0.85；原 seed 牙面不整体刷白。EMA 为 `0.65 current + 0.35 previous`，再次乘当前 fill，旧帧 RGB 从不复制。
5. 仅将稀疏 delta/alpha 映回当前原始完整画幅；nearest hard allowed 再裁剪支持，并在原始当前画幅再次排除红色像素与已有浅色牙面像素。最终 uint8 原帧复制后只写允许区，不把重采样整图作为输出。检查允许区外最大像素差为 0。

**局限：**这是亮度/颜色及行结构启发式，不是可靠皮肤/牙齿语义分类；中性浅肤、反光或不恰当 ROI 仍可能误判。缺少可见下牙时有意回退。partial affine 不能解释所有三维运动，牙面原样保护首先成立于对齐 patch，亚像素映回及 hard mask 的语义准确性仍必须远程逐帧检查。增加亮度或填缝不等于更自然；不改变牙冠整体高度、不生成遮挡牙、不保证用户要求的整排平缓外观。当前候选范围未经人工批准，不用于训练或正式发布。

## 证据产物

- `before/0000.png`、`after/0000.png`、`allowed/0000.png`：连续选段的每个真实解码完整画幅及当前候选许可。PNG 序号与原视频帧号映射见 metrics。
- `before.mp4`、`after.mp4`、`compare.mp4`：相同完整选段、真实输入 fps（不可用或超出支持范围则拒绝，不虚构帧率）、无声；奇数尺寸只在右/下边补齐偶数编码尺寸，不补帧。输出均再次完整解码核对帧数。
- `metrics.json`：输入 SHA256、Git HEAD codeSHA 与实际代码文件 SHA256（避免未提交代码被误记为 HEAD）、全部参数、逐帧原因/changed/allowed/outside0，`source-role=final-canvas`，`mask_approved=false`、`training_enabled=false`、`temporal_validated=false`。绝对本机路径可能在参数中，报告未经脱敏不得上传。
- 区外零差只针对保存前无损帧，不延伸到有损 MP4。视觉收益需要正常尺度完整对照与连续关键窗口审阅，不能以 changed 数证明改善。

## 本地验证

```text
python -B -m unittest discover -s tests -p test_upper_gap.py -v
```

本次 Windows 本地执行：6 项 NumPy 测试通过，1 项 OpenCV tracker 测试按平台跳过。覆盖黑缝变浅、seed/下牙/红唇与区外保护、空/单亮带/闭口/肤色回退、EMA 当前支持与重置、零强度（即使已有EMA也清状态且完全回退）及 dtype/shape/非法数值。

尚未执行远程 OpenCV、真实视频、ffmpeg 或视觉验证。远程验收须补跑 tracker 测试与真实选段，检查失败比例、每帧 outside0、完整编码帧数、遮挡边界和自然度；任何非零区外差或误填唇/下牙均不得采用。
