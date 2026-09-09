# 上牙显隐：嘴部纵向响应诊断开关

本实验针对用户批准的 9–11 秒上牙显隐问题。连续检查发现开口偏小，尚未证实存在独立“掉牙”。这是诊断候选，不代表已优化成功，不合并 v1。

## 参数与算法

CLI：`--lip-vertical-gain 1.1`；默认 `1.0`，仅接受有限数值 `[1.0, 1.25]`。新增字段同时存在于 ArgumentConfig / InferenceConfig，沿用入口现有字段传递机制。

在原逻辑构造 `delta_new` 后、世界坐标 `x_d_i_new` 构造前，仅修改唇部索引 `[6,12,14,17,19,20]` 的 y 分量：

```text
delta_new[lip, y] = source_exp[lip, y]
                   + gain * (delta_new[lip, y] - source_exp[lip, y])
```

此处 y 是表达空间分量，不是屏幕像素纵轴，也不是显式上牙结构。正负增量均按比例放大，不能保证每帧开口都增大。x/z、非唇点、原旋转、scale 和平移计算不改；首帧增量为零时不引入偏移。默认 1.0 不调用算术 helper，保留原计算路径。

## 支持范围与拒绝条件

非默认值仅支持人像照片源 + 实际视频驱动 + relative motion，animation_region 为 all/exp/lip。源视频、图片驱动、模板驱动、非 relative、eyes/pose 区域以及 eye/lip/source-video-eye retargeting 均在 execute 加载媒体前报错，不静默忽略。该校验发生在 pipeline 初始化之后，不保证在模型加载前拒绝。

普通 lip retargeting 在 relative 分支会用 `x_s + eyes_delta + lip_delta` 替换原运动，丢失原头眼运动，因此本实验不使用该路径。原嘴唇归一化、stitching、expression-friendly 的运动倍数及最终 driving_multiplier 路径保持不变；做对照时这些参数必须固定。

## 边界与验收

- 开关应用于输入的全部帧，不内置 9–11 秒时间门控；该时间段是重点评测窗口。
- 只限制表达空间修改，不承诺输出像素区外不变。共享 W/G、stitching 等可能引发头眼、嘴唇、身份或其它区域回归。
- 尚未运行 Tensor、CUDA 或模型验证，也未验证视觉收益。NumPy 与 AST 测试不能替代真实推理。
- 后续获准远程实验应保持代码、输入及其余参数一致，对比默认与小幅增益，检查完整输出及 9–11 秒连续帧的上牙显隐、唇形、头眼运动和全局回归。无可靠收益则不采用，不发布新正式版本。

本地允许的检查（不导入 pipeline / torch）：

```text
python -B -m unittest discover -s tests -p test_lip_vertical.py -v
```

覆盖默认输入不变、非唇和 x/z 精确不变、增量比例、零增量、空/非法参数与形状、拒绝模式、配置默认值及 AST 接入位置。helper 仅依赖标准库，NumPy 用于测试；Tensor 分支使用 clone 与相同索引接口，未实际执行 Tensor 测试。
