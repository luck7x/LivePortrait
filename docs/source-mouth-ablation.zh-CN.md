# 源图 × 嘴唇归一化与固定学生探针

本阶段只比较老高清闭嘴源、新露上牙源各自的 v1 normalize_lip on/off，并在新图 on 的同一 F/xs/581 K 上测试17历史学生固定 strength=.5。不训练、不挑强度、不自动扩展 clip80；换源图不是纯牙齿像素消融。代码实现不代表效果已通过，最终须完整视频与独立图像审阅。

## 入口与预算
- `scripts/probe_source_mouth.py --stage off|student --workspace <root> --snapshot <verified-on-snapshot> --budget-root <shared-new-budget> --output <new-output> --authorize-source-mouth`。
- student 另须 `--checkpoint <historical-decoder.pt>`；off 拒绝 checkpoint。无 labels/GT/训练数据参数。只接受本轮两张已知 SHA 的源图，student 只接受新图及已核验固定 checkpoint SHA。
- 输出不可覆盖，源码/snapshot/预算目录分离；三个条件使用同一个预算根。每阶段600秒，worker585秒截止预留审计；supervisor 实际墙钟含失败累计≤1800秒。成功条件禁止重跑；不自动额外 trial。
- 本轮累计新增输出≤512MiB（保留32MiB余量）、project≤20GiB，按实际分配空间计算。预算不足停止，不删旧文件、不安装。不可与其它不遵循本入口锁的写入并发。
- Linux、干净 Git、`PROBE_CODE_SHA` 固定完整SHA、单GPU UUID、至多4CPU affinity/线程。父代理启动前重查空闲卡及用户总额度≤2；此次不授权启动GPU。
- 继承项目内 HOME/TMPDIR/TMP/TEMP/XDG_CACHE_HOME/TORCH_HOME/HF_HOME/CUDA_CACHE_PATH，禁止全局改变；必须 `PYTHONDONTWRITEBYTECODE=1`、`IMAGEIO_FFMPEG_NO_PREVENT_SIGINT=1`。子进程固定 CUBLAS 确定性设置，复用原环境，不下载模型。
- 复用 `OwnedProcessGroup` 的 waitid/WNOWAIT：保留 leader 未回收以避免 PGID 复用，只清理当前监督进程组。running/损坏的旧 supervisor 阻止继续，不能将失败时间清零。

## 运动与学生路径
- `prepare_contact_samples.load_inputs` 先核验原snapshot文件、F/xs/source_input/581K、权重及W/G代码；再核验当前 pipeline/wrapper/运动/retarget/pasteback相关原blob，不另复制运动公式。
- driving_motion.npz 实际有3488项：581×6个 motion 数组＋eyes/lips两个列表数组，超出旧 bounded_npz 的64项上限。新增有界读取检查确切键/shape/float32/NPY长度、4MiB压缩/展开上限及 report.template 每数组哈希，不读取 pickle。
- off 借用原 `LivePortraitPipeline.execute` 的模板分支：只读 verified 数值模板；source canvas/crop/source_input/F 固定，绕过 Cropper 实例/检测；原M仅对固定 source_input 调用一次并核验 xs，on/off共用同一 source info。
- 先完整收集 on 的581个最终K并与已有snapshot精确比较，再只关闭 normalize_lip 收集 off K。全部K经过原 pipeline 原公式与原 stitching；不从最终K手减delta。收集阶段截获 warp_decode 后只存K，在第581帧停止以阻止pipeline写媒体/原driver pkl。最终视频独立流式重放这些K。
- off 的最终K允许变化、不能声称K固定。on的原W/G raw/full 回归覆盖18个既定时刻；student 的原baseline raw/full 则581帧全部精确对照，student始终使用新图原on K副本。
- student 复用 RealTeethDecoder、native pre/up_1特征、预测gate、原FP16 compose；固定.5，无GT/mask/帧号patch输入。核验原owner报告/supervisor、checkpoint/metadata哈希、原训练代码blob、五原权重及其它原模型文件；`weights_only=True`、严格 delta state 校验。以blob兼容允许新实验代码SHA，不伪改旧metadata或旧load_checkpoint的SHA检查。
- 所有原模块冻结/eval，原W/G同时走production FP16；记录参数/buffers、五权重、输入/代码前后哈希与学生不变。CUDA时间线含CPU等待，不冒称纯kernel耗时。标签语义保护与完整牙列质量仍待独立审阅。

## 完整视频与验收
- 每条件生成 `candidate_silent.mp4`、原音轨streamcopy的 `candidate_full.mp4`，及左既有on A0/右候选的 `compare_full.mp4`；581帧、25fps、23.24秒、CRF18/yuv420p。不烧列名，父页面须标注。两份on直接引用既有A0，不复制。
- 三个新候选分别为 old-off、new-off、new-on-student-.5；不能将学生片写成new-off。比较视频各panel等比缩放/padding512，保持连续原帧，不循环短片、不加速、不使用-shortest。
- 既有A0及各新视频完整解码581；有声文件逐包hash/PTS/DTS/duration/流元数据与原driver一致。每条件存18组raw对比PNG和最终K，不存全部PNG/H。
- 每帧报告K/raw/full哈希及是否变化；off在18个回归点另计像素变化，不能把稀疏像素计数称全帧像素评估。student全581另验证predgate/ROI外float与uint8零差及pasteback传播域外零差；这不等于语义只改牙齿。
- 所有 `quality_pass=false`，等待父代理完整视频核验与唯一图像审阅会话；无收益不采用，不自动训练/扩实验。旧学生已知不达标，固定.5是诊断条件，不是最优强度。

## 本地验证与待查事实
`python -B -m unittest tests.test_source_mouth -v`：仅标准库/NumPy/AST，实际子进程导入防火墙确认无Torch/CV2。本地禁止模型执行；CUDA数值回归、源M到xs跨运行精确性、600秒编码可行性必须远程实测，任一回归失败即停，不能放宽阈值凑结果。
只读核验已确认两snapshot关键文件与原manifest一致、581 motion六字段完全相同、原驱动M输入hash和音轨一致；但eyes/lips列表581帧均有差异。两图必须各用其自有模板，不互换。当前明确关闭眼/唇retarget，差异对本条件的实际影响以on全K回归为门槛，不自行归因。
本地私有操作记录位于工作区 `results/3090/source-mouth-ablation-v1/`，含服务器路径的JSON不随源码发布；本脚本/文档无私有路径或连接信息。此实现只经本地检查，尚未运行三个新视频，不宣称已完成优化。
