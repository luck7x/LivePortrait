# 新JPG三版本：仅本地准备

基点a0c9c5c，分支experiment/new-jpg-three。原ded9源继续允许；新增受控源SHA：cec1314cb81c43a476ba827779e5b0ee4888753fdca6c881cbaf739c766ed759。原图959×1280、RGB JPEG，不修改原文件。

仅在本补偿脚本内新增validate_source_config，保留原base.check_config的全部配置门槛：normalize/stitch/relative/FP16/do_crop为True；eye/lip retarget、torch compile、source-video eye retarget、crop-driving flag为False；multiplier=1、animation_region=all、driving_option=expression-friendly。共享probe_source_mouth.check_config不改。

报告source_sha256取真实report.inputs_before[ArgumentConfig.source]，不再硬编码ded9。算法、alpha1全581K回归、独立delta存储及原stitching+scaled lip delta后multiplier顺序、原预算不变。若新源比值低于原归一化阈值，V1/V2可能相同，不强造差异。

待资源可用后，仅clip118：纯官方默认False、项目V1 alpha1、V2 dynamic三份完整581帧/25fps/23.24秒原声视频。三列顺序同上，画面不烧录名称/文字、不加顶部标题条，仅页面解释。历史视频、标签、原权重不改。

当前尚未启动本轮模型，没有新视频。用户最新确认本人无其它GPU任务，允许余量充足时共享单卡；不能再把全服务器忙卡数等同用户额度。每次启动前查询memory.free，优先余量≥12GiB且负载较轻的单卡，不操作他人进程。本阶段没有服务器操作、GPU/本地模型运行、部署、commit/push或删除。请求与原图metadata见工作区results/3090/new-jpg-three-v1/request.json。本脚本新增窄路径适配：仅合法补偿stage，将参数浅副本交原严格student同级布局校验；不改变原参数、实际推理stage或ledger，不启用学生。允许同预算根的不同直接同级snapshot/output，保留输入输出互不包含、反逃逸、源码分离和新输出要求。适配由ExitStack恢复，共享helper文件不改。官方/V1/V2仍须同预算总账，不能更换预算根隐去成本。

本地测试入口：python -B -m unittest tests.test_mouth_compensation -v；只使用标准库/NumPy与协议替身，不将其当CUDA验证。
