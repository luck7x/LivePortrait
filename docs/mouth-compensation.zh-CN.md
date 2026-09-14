# 源口型补偿探针：阶段1实现

## 范围与原始算术顺序

独立分支 `experiment/mouth-compensation` 从88f95c6建立；只新增探针、测试和本文，不改src/或主线。固定20的16:46新源图（SHA ded9d6ed7facbf5bf2646e982858d87ffc7ebd45b110a95706df60157f3fbed2），复用clip118-on/clip80-on各自的源snapshot和驱动template。不叠加17学生、不训练、不读取GT/人工目标帧或做RGB贴补。

**纠正此前口头描述：** 当前启用stitching的原pipeline实际为

```text
x_d_i_new = stitching(x_s, x_d_i_new) + lip_delta_before_animation
x_d_i_new = x_s + (x_d_i_new - x_s) * driving_multiplier
```

不是“先加lip delta再stitch”。探针保持这个原顺序不动；缩放的是原MLP返回的独立新tensor，由原pipeline按上式消费，绝不在最终K后手减delta或重写运动公式。

## 预登记候选

clip118仅三种 `--mode`：

- `fixed-half`：alpha=.5。
- `fixed-three-quarter`：alpha=.75。
- `dynamic`：`alpha=1-.5*smoothstep(clamp((r_t-.03)/(.25-.03),0,1))`，smoothstep(z)=z²(3−2z)。

r_t来自该snapshot真实 `c_lip_lst`，阈值.03/.25、下限.5预先固定，不看输出再调。无EMA、无人工时间表。只有原source ratio≥原lip_normalize_threshold时归一化分支实际生效；否则不强造delta。报告区分请求alpha与normalization_active。

每次调用先以alpha1重建全部581K，要求与20原on cache精确一致，再收集一个候选。源crop/canvas/source_input/F/xs固定；源M只算一次、精确检查xs，两遍复用source info。每遍MLP原输出detach/clone后再clone出独立mutable tensor，检查与源/模型/原输出storage不别名。初始化alpha[0]，本帧K复制完成后才更新下一帧alpha，f580后不再越界更新。原MLP输出和所有模型参数/buffers不变。

render复用现有原W/G production FP16、原pasteback和流式FFmpeg；对原on的18个固定raw/full帧作精确回归，生成完整581帧、25fps、23.24秒候选及左右对比、原音轨包/PTS/DTS验证，保存alpha/K和少量raw_pair。报告明确mouth-compensation，不叫off；源口型干预可能影响嘴唇/下牙/其它区域，没有语义区外零差保证。

## 接口与监督器兼容

```text
python -B scripts/probe_mouth_compensation.py
  --workspace <project> --snapshot <对应20的on-snapshot>
  --budget-root <本轮共享预算根> --output <不存在的新目录>
  --driver clip118 --mode fixed-half
  --authorize-mouth-compensation
```

其余两种mode同接口；每次只运行一个候选，不自动遍历。clip80必须由父根据clip118结果选一个mode，再显式传 `--driver clip80 --selected-for-clip80`；已有成功clip80候选后禁止扩跑其它mode。

复用 `probe_source_mouth.main` 的进程级监督器：新Parser提供它要求的共有参数、checkpoint=None及唯一stage `mouth-compensation:<driver>:<mode>`，授权开关映射到其authorize_source_mouth字段；没有student stage。`ExitStack` 临时适配parser/worker/__file__，保证子进程启动本新入口并识别隐藏 `--_worker`，正常/异常退出均恢复。运动hook使用另外的ExitStack，每个case独立，无跨case残留。

输入snapshot在其它task，必须与新budget-root分离；新output是其严格子目录。保留Linux/单GPU UUID、4CPU、固定干净SHA、缓存隔离、ImageIO继承进程组要求。每阶段600秒（worker585秒留审计余量），累计≤1800秒，新输出≤512MiB、project≤20GiB；原OwnedProcessGroup/锁/失败计时规则不变。官方CLI及素材准备/音轨恢复也必须计入本轮同一预算与真实supervisor ledger，不把失败算0。GPU0只读检查时已有未知占用，不读取其进程，不保留其他卡。

## 官方默认CLI单独落地

父后续另建纯官方9b294工作树，用官方inference.py，仅传source/driving/output，不能hook、改src或强制normalize on。该版本ArgumentConfig默认normalize_lip=False；仍须实际执行，不能把历史off充作官方结果。官方默认预处理/运行环境与缓存定量实验可能不同，不宣称官方片与本探针alpha0 raw精确公平；本探针定量主对照是同源alpha1。

为避免官方音频-shortest路径截帧，可先在本轮inputs保存原驱动video-stream-copy无声副本，核验视频流packet和完整581帧RGB hash一致，再交给原CLI。保留官方silent原产物，另独立stream-copy原音轨并核验581/25fps/23.24秒、audio packet hash/PTS/DTS。驱动旁的官方pkl只允许落在本轮inputs，不触碰原素材。官方音轨保全过程单独标记，不伪称纯CLI直接输出带声片。本阶段不编写/执行庞大官方operator，具体命令与计费由父审核后确定。

## 验证与仍未验证项

本地 `python -B -m unittest tests.test_mouth_compensation -v`：6项通过，覆盖固定/动态端点与无EMA滞后、581步cursor顺序、独立storage/原值保护、Parser与监督器适配/异常恢复、clip80选择及单候选限制、原stitch顺序AST和独立进程导入防火墙。fake tensor仅是NumPy协议替身，不冒称Torch/CUDA测试；本地未导入Torch/CV2执行模型。

只读preflight见工作区 `results/3090/mouth-compensation-v1/preflight.json`：两份20snapshot关键文件及原权重实际SHA一致，源source_input/F/xs一致，真实嘴唇序列581项；项目19,951,329,280字节。后续代码worktree/缓存也要计入project预算，不把旧余量永久使用。

尚未进行真实CUDA alpha1/K回归、delta存储验证、完整候选生成、官方CLI执行或视觉验收。已有代码和CPU协议测试不代表优化成功；任一真实回归失败须停下，不放宽容差。阶段1无GPU/远程写入、commit/push或旧证据删除。
