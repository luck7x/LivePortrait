# 牙齿局部处理共享契约（B/C）

本契约是 B（局部细化）与 C（局部结构/时序机制）共用的像素保护和真实数据准入底线。它不实现训练、牙区自动推断或视觉审核。

## 像素保护

`teeth_local.compose_allowed_region(base_rgb, candidate_rgb, allowed, alpha, protected=None)` 接受：

- `base_rgb`：固定 v1 底片，严格为 `(H, W, 3)` 的 `numpy.uint8` RGB。
- `candidate_rgb`：同尺寸、同 dtype 的候选 RGB。
- `allowed`：同空间尺寸的 boolean 人工审核下牙区域；`None` 等价于全空区域。
- `alpha`：同空间尺寸的浮点数组，所有值必须有限且位于 `[0, 1]`；`None` 表示全零，即不启用修正。
- `protected`：同空间尺寸的 boolean 上牙、嘴唇等保护区；`None` 等价于全空区域。

`allowed` 与 `protected` 有任何重叠时立即拒绝；`alpha > 0` 的范围不得超出 `allowed`。实现先复制底片，只对 `alpha > 0` 的像素按
`round(base * (1 - alpha) + candidate * alpha)` 写入（NumPy `rint` 舍入）。因此 `alpha == 0` 以及 `allowed` 外保持 v1 底片 bit-exact；函数不修改任何输入。空 `allowed` 或全零 `alpha` 返回内容不变的底片副本。

函数返回 `(output_rgb, ProtectionStats)`。统计中的：

- `outside_max_diff`：输出与底片在 `allowed` 外所有通道的最大绝对差，合格值必须为 `0`；
- `changed_pixel_count`：与底片任一 RGB 通道不同的像素数，不是通道数，也不等于 `alpha > 0` 数量。

`validate_protected_output` 可独立验收已有输出；发现 `allowed` 外或 `protected` 内发生变化会拒绝。最终 MP4 的有损编码差异必须与本无损帧检查分开统计。

`allowed` 必须来自明确的人工审核分割。矩形 ROI、检测框、关键点包围盒或程序自动预测不能被记作“已审核牙区”；本包也不会自动推断或扩张牙区。

## 真实数据最小准入记录

`validate_data_record(record.json)` 只检查 JSON 声明、引用文件存在性及 SHA-256 完整性。最小格式如下：

```json
{
  "contract_version": "1",
  "source": "数据来源说明",
  "authorization_scope": "授权用于何种训练/实验的范围",
  "training_approved": true,
  "mask_reviewed": true,
  "paired_gt_available": true,
  "mask": {"path": "relative-mask-path", "sha256": "64位小写十六进制"},
  "paired_gt": {"path": "relative-gt-path", "sha256": "64位小写十六进制"}
}
```

记录最大64KiB，contract_version必须是字符串`1`。只接受以JSON所在目录为基准的相对文件路径，解析后的真实路径也必须留在该目录内，禁止`..`或符号链接逃逸。调用方仍须核验JSON记录本身位于授权的数据工作区。三个布尔字段必须明确为JSON `true`；字符串必须非空；mask与paired GT文件必须存在且哈希匹配。

该函数**不**核实授权声明的法律有效性，不做素材内容、人物身份、mask 质量或 paired GT 对齐的视觉核验，也不把通过程序校验等同于允许训练。

父代理在公共预处理/训练放行前仍需人工审核：

1. `contract_version` 是否为本轮 B/C 采用的版本；
2. `source` 与 `authorization_scope` 的原始依据及是否覆盖本次训练用途；
3. `training_approved` 的批准主体和范围；
4. `mask_reviewed` 对应的审核人、审核标准，且确为分割而非矩形 ROI；
5. `paired_gt_available` 的配对关系与视觉对齐；
6. 两个文件路径是否指向本轮固定数据，SHA-256 是否已纳入实验记录。
