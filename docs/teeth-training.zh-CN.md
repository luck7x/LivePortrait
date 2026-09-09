# B 局部训练基础与 r002 structure 目标（新目标尚未运行）

本实现只增加独立局部残差网络，不修改 `src/`、`inference.py` 或 v1 权重。代码存在、CPU 契约通过、GPU 合成 smoke 通过、真实训练完成、视觉改善是五个不同状态，不能相互替代。当前仅允许本地 AST 和 NumPy/schema 检查；GPU 前反向及真实训练需在已获准 Linux 服务器另行执行。

## 数据契约

沿用 [共享契约](teeth-local-contract.zh-CN.md) 的许可、人工 mask 审核、配对 GT 声明与 SHA-256 门槛，并增加：

```json
{
  "contract_version": "1",
  "source": "素材来源及配对关系的真实依据",
  "authorization_scope": "实际批准的训练用途和范围",
  "training_approved": true,
  "mask_reviewed": true,
  "paired_gt_available": true,
  "dataset_id": "reviewed-sequence-unique-id",
  "native_reference_confirmed": true,
  "mask": {"path": "pairs.npz", "sha256": "替换为实际64位小写SHA256"},
  "paired_gt": {"path": "pairs.npz", "sha256": "替换为实际64位小写SHA256"}
}
```

`native_reference_confirmed=true` 必须表示人工已确认 `base_rgb` 为固定 A/v1 底片，训练参考直接取该序列 `base_rgb[0]`，不是 GT 或混入 GT 的外观参考。这是声明而非程序能证明的来源事实。GT 只进入监督损失和误差统计。未经真实授权、配对/对齐及牙区审核，不得为了通过校验填写 true。

`paired_gt` 为 NPZ，严格包含以下六个键：

| 键 | dtype / 形状 |
|---|---|
| `base_rgb`, `target_rgb` | uint8 `(N,H,W,3)`，RGB |
| `allowed`, `protected` | bool `(N,H,W)` |
| `alpha` | float32 `(N,H,W)`，有限 `[0,1]` |
| `frame_ids` | 整数 `(N,)`，非负、严格递增，原视频帧号 |

`1<=N<=96`、`1<=H,W<=256`；allowed 不得与 protected 重叠，alpha 非零支持域必须位于 allowed。序列必须存在有效 alpha>0 区域，但允许闭嘴/不确定帧全空回退。审核 mask 可与配对 NPZ 是同一文件；若独立，必须为恰含 `allowed/protected/alpha` 的 NPZ，dtype/数值与配对 NPZ 完全一致。不得用矩形 ROI 冒充审核 mask。

记录必须位于授权 workspace 内，引用文件必须位于记录所在目录内，解析符号链接后仍须满足约束。压缩文件大小与 ZIP 声明解压总大小均不超过 128 MiB，不写磁盘解压；读取前核对 NPY header、实际 payload 大小和维度，拒绝 object、结构化、Fortran 数组及额外 ZIP 条目。所有加载使用 `allow_pickle=False`。

train/val 必须分别提供准入记录，`dataset_id` 不同且 paired NPZ 哈希不同。这只阻止明显重复，不保证无内容泄漏，**不宣称身份级划分**；人工仍须核对源视频、人物与时间段关系。

## 网络、训练与导出

- `LocalResidualB.forward(base, reference, allowed) -> residual`：base 为 float `[0,1]` 的 `(B,T,3,H,W)`；reference 为 `(B,3,H,W)`；allowed 为 bool `(B,T,1,H,W)`。
- 每帧共享 6 输入通道（base+固定参考）、宽 24 的三层 3×3 CNN，ReLU，末层权重与偏置零初始化，tanh 残差；初始输出不改变底片。
- `--variant C` 才懒导入 `model_c.TemporalResidualC`。当前 B 分支不提供 C 实现；选 C 会明确失败，不自动回退到 B。CPU 导入 dataset/training/CLI 不导入 Torch，模型模块仅在准入和运行环境检查后加载。
- 每步整个训练序列、batch=1；固定 seed `20260909`，Adam lr `1e-4`。验证序列不参与训练，不挑选最佳权重，不做身份级泛化声明。
- 可微合成：候选为 `clamp(base+residual,0,1)`，active=`allowed & (alpha>0)`，`where(active, base*(1-alpha)+candidate*alpha, base)`。非 active 像素严格保留 base；alpha 和 mask 不学习。
- 默认 `--objective pixel` 保留原残差、原始 GT 的允许区 alpha 加权 L1 与 `1e-4` 残差平方正则，保留旧训练数学路径。`--objective structure` 见下文；无全图训练损失，无原 v1 模型参数更新。
- 最终候选先量化为 uint8，再调用现有 `compose_allowed_region` 用原 alpha 合成，逐帧验收 outside 最大差为 0，额外检查 alpha=0 不变。这与浮点训练存在明确的量化误差，不用二次 alpha 合成，也不宣称与浮点输出逐位一致。

真实训练成功才写一个 `checkpoint.pt`、一个 `predictions.npz` 和 `metrics.json`。NPZ 包含 `train_rgb/val_rgb` 及各自 `train_frame_ids/val_frame_ids`，保存两段全部准入帧的 uint8 输出，不拼成原始整片、不附原音轨、不生成 MP4。指标记录原始 A / 训练后 `[0,1]` 尺度 alpha 加权 L1、逐帧及总变化像素、outside0、SHA/实际代码文件哈希/dirty 状态、数据哈希、步数、耗时、峰值 GPU 分配字节。低像素误差不是牙齿视觉改善证据。

## r002：抑制增白捷径的 structure 目标

首轮 200 步真实训练盲审主要看到中央增白亮带，而非可靠牙缝恢复；低 L1 不代表成功。此次只新增可选训练目标，不改模型容量、mask、alpha、GT 文件或 A/v1 主体，不增加时序损失。用户允许沿用同一 mask 继续训练，不等于此目标已经验证有效。

选择 `--objective structure` 后：

1. **训练、评测和未来推理共用** `objectives.prepare_residual`：按每帧、每 RGB 通道计算 allowed 内 alpha 加权残差均值并减去；非 active 区归零。若最大绝对值超过 1，用该帧该通道单一比例缩放，不能逐像素裁剪残差。空 mask 返回零。此步骤不接收 GT。
2. 仅在损失内将 GT 按同一权重的 RGB 均值平移到 A 的原色均值，再 clip 到 `[0,1]`；不改写原始 GT。裁剪后均值不保证仍严格匹配。
3. 损失为调色 GT 的既有 alpha 加权 L1 + **未调色原始 GT** 的梯度匹配 + `0.5 × 最终浮点合成输出相对 A 的局部平均 RGB 色偏` + `1e-4` 后处理残差平方正则。梯度为水平/垂直有符号一阶差分的 L1，只有相邻两点均 active 才监督，边权取两点 alpha 的较小值，按有效边权×3 归一化；固定区边界不当牙缝，空邻域返回可微零。色偏先对每帧每通道按 alpha 求有符号均值，再取绝对值，对非空帧和 RGB 通道等权平均（不让不同帧正负抵消）。

残差零均值不保证最终输出零色偏，因为候选 RGB 截断、alpha 合成和 uint8 量化仍可造成偏移，故另加最终输出色偏项。梯度目标也不保证能恢复不可辨认牙缝，须用同输入完整窗口盲审验证，不能只报训练 loss。

评价保持 `a_weighted_l1/after_weighted_l1` 对**原始 GT** 的 RGB L1；新增 `a_raw_gt_gradient_error/after_raw_gt_gradient_error` 对同一 raw GT 的梯度误差，以及 `a_mean_rgb_shift_from_a/after_mean_rgb_shift_from_a` 局部平均色偏（A 自身为 0）。这些指标均基于最终量化、原 alpha 合成的交付帧；不拿 pixel 与 structure 的训练 loss 数值直接比较。

checkpoint 和 metrics 均记录 `objective`，`code_hashes` 包含 `teeth_local/objectives.py`，运行前后逐文件哈希一致性检查保留。**未来加载 checkpoint 推理必须先调用 `verify_objective_metadata(checkpoint, repository)` 校验 objective 和代码哈希，再将 model.forward 的原残差交给 `prepare_residual(torch, residual, allowed, alpha, objective)`，之后才做候选 clamp 和原保护合成；禁止仅裸 model.forward 相加。** 当前不新增独立 checkpoint 推理 CLI。缺 objective/objectives.py 哈希的历史 checkpoint 不可静默当成 structure；此校验器明确拒绝，需要走经确认的历史 pixel 代码路径。

## 运行门槛（下列为未来获准远程执行示例，不是本地运行建议）

必须先依工作区规范读取服务器操作文档、确认现有环境、数据人工准入、GPU 空闲和本用户总占用不超过两张。本程序单进程仅接受一个 `CUDA_VISIBLE_DEVICES`，不自动找卡、不占第二张卡、不停止他人进程、不安装依赖。单卡可见性检查不能证明物理 GPU 空闲或全用户 GPU 配额，需启动前人工检查。

工作区必须已存在、包含当前代码工作树，且由操作者确认属于已授权范围；`--authorize-workspace` 仅是已有授权的显式确认，不授予新权限。输出必须是 workspace 内尚不存在的新目录，父目录须已存在。上限：1000 步、1800 秒、workspace 20 GiB（启动预留 128 MiB 输出），时间在每步和评测边界复查，磁盘每30秒及保存前复查。时间包含数据加载；同步边界超过预算即失败，不写成功报告。单次 CUDA 操作无法由 Python 硬实时中断，严格墙钟截止仍需授权的作业调度/外部超时；不要把该同步检查说成硬实时抢占。

在现有环境和已选定单 GPU 下，从代码目录运行：

```text
python -B scripts/train_teeth.py --workspace <workspace> --output <workspace>/runs/B-new --authorize-workspace --train-record <workspace>/data/train/record.json --val-record <workspace>/data/val/record.json --variant B --steps 100 --seconds 1800
```

目录禁止覆盖。失败时可能留下不完整新目录；没有 `metrics.json` 不能视为成功，不自动删除产物。记录 SHA 和工作树文件哈希不能替代运行期间禁止改代码/数据的操作纪律。workspace 总量检查不是多进程磁盘配额，其他作业仍须共同遵守已授权预算。

### 独立合成 smoke

使用同一 Linux/workspace/单 GPU 门槛，`--synthetic-smoke` 不接受 train/val 记录，内部固定随机合成 3×16×16 RGB 张量执行两步前反向与保护验收。只输出标记 `not_real_training` 的 `metrics.json`，不写 checkpoint、真人预测或正式权重。它不证明真实牙区准入、训练有效或视觉改善。

```text
python -B scripts/train_teeth.py --workspace <workspace> --output <workspace>/runs/B-smoke-new --authorize-workspace --variant B --synthetic-smoke --seconds 120
```

## 本地允许的验证

```text
python -B -m unittest discover -s tests -p test_teeth_protection.py
python -B -m unittest discover -s tests -p test_teeth_training_contract.py
```

本地只做 AST、schema、NPZ、NumPy 算术与禁止 Torch 导入的隔离进程检查。新增 `tests/test_teeth_objectives.py` 数学测试仅当 Linux 且显式设置 `TEETH_TENSOR_TESTS=1` 时在 Torch CPU 执行；Windows 无条件跳过且不导入 Torch。测试覆盖加权零均值/统一缩放、空 mask、亮度常量偏移的梯度不变性、可微梯度、固定边界/孤立像素忽略及 GT 不变。

获准服务器现有环境可执行 `TEETH_TENSOR_TESTS=1 python -B -m unittest discover -s tests -p test_teeth_objectives.py`；这不是本地执行许可。新目标正式运行须显式添加 `--objective structure`，未指定仍为 pixel。不能以 AST/NumPy 检查宣称已完成这些数学测试、CUDA smoke 或真实训练。服务器不可用也不得改用本地模型运行。
