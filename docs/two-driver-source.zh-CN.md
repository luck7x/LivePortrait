# 同一新源图 × 两段驱动：阶段1工程适配

## 固定范围

从 `1d396b5a0acea258b351da611c45888e64b35794` 建立独立 `experiment/two-driver-source`。
源图SHA256：`ded9d6ed7facbf5bf2646e982858d87ffc7ebd45b110a95706df60157f3fbed2`，1254×1254、RGB PNG、2,373,827字节。只读核验原文件，未生成/编辑照片；私人素材路径仅记录工作区results metadata，不放进公开脚本。

目标为该源图分别由clip118、clip80驱动，每路原v1 normalize on＋17历史学生固定0.5，共4个完整视频、2个同步对比。不训练、不跑新图normalize off，不改src/原F/M/W/G或历史checkpoint。学生此前未采用，本次只是固定对照，不预告改善。

## 最小新增接口

`scripts/snapshot_upper_teeth.py` 保留原pipeline生产流程，增加：

- 驱动准入 `check_driver`：581帧、25fps、23.24秒、正尺寸、max(width,height)≤1280；不再限定1024方形。`check_video(square=True)`旧公共接口仍保留，其它既有使用者不受影响。
- 非方驱动仍由**原pipeline自动crop_driving_video**后调用原M制作自身template；不先强缩方、不借用clip118的driver模板。报告 `actual_driving_auto_crop` 记录实际hook是否调用，并核对非方路径。
- `--source-cache <首路成功snapshot目录>`：只返回source_input/F/xs、crop和source canvas；不返回/使用缓存finalK或driver模板。来源source_snapshot内旧finalK仅作原文件完整性验证，绝不传给第二路推理。
- 缓存必须同代码SHA、report+supervisor均completed、supervisor returncode0、原输入/权重/参数状态前后一致；核对新source文件SHA与缓存原source记录相同，所有source文件/数组hash匹配，路径不能逃逸或与新output重叠。
- 第二路 `crop_source_image` 直接返回缓存crop，不检测source；原pipeline仍执行prepare_source，保持真实打包/stride，取F前对其值/hash严格核对，再提供缓存F。源M照原pipeline运行，warp前xs必须与缓存**精确相同**；不能用容差或旧K替换绕过失败。
- 当前输入canvas与缓存canvas严格相同，输出crop/矩阵保留原数值；报告 `source_cache.source_arrays_exact`、cache files前后hash。两路driving crop/M/运动模板/finalK独立正常执行。
- `--max-new-mib 512` 把当前进程snapshot累计新产物上限缩到512MiB，默认1024保留历史接口；本轮两个snapshot都必须显式传512。原64MiB余量与project20GiB上界保留。

`scripts/probe_source_mouth.py` 中student只准入上述新源SHA，仍固定历史checkpoint SHA `32e108b4258cd7c5c6fcf4b4a09cd1e20369a8cddc3183821106d841ac6c2b2c`、strength=.5、threshold=.95。原meta不改、原blob兼容及严格weights-only加载不变；新源off被拒绝。旧1d396代码及已完成实验不变。

## 父代理后续执行顺序（本阶段未运行）

1. 首路snapshot：同一源文件＋clip118，现有全部CLI参数，追加 `--max-new-mib 512`，新输出如 `<budget>/clip118-snapshot`。
2. 第二路snapshot：同一源文件＋clip80，另加 `--source-cache <budget>/clip118-snapshot --max-new-mib 512`，新输出如 `<budget>/clip80-snapshot`。
3. 对两份新snapshot分别调用probe的 `--stage student --checkpoint <历史decoder.pt>`，各新输出目录。不要直接用旧run_real_teeth的硬编码cross入口。

四阶段共用**同一**全新budget-root。snapshot与student均用 `source-mouth.lock`；snapshot supervisor新增真实status与累计墙钟，复用elapsed_charge，因此总计包含两个snapshot＋两个student及失败。每阶段600秒（worker585秒，预留审计），累计≤1800秒；不足完整600秒余量不启动。旧supervisor不追改，不覆盖旧输出。snapshot可变report/supervisor不再进入files自哈希清单。

运行仍须单GPU UUID、用户总额度≤2、4线程、干净固定PROBE_CODE_SHA、隔离缓存、`PYTHONDONTWRITEBYTECODE=1`、`IMAGEIO_FFMPEG_NO_PREVENT_SIGINT=1`；重新查空闲卡，不全局安装或修改环境。四阶段的实际显存/墙钟/空间及音视频均须远程验证，不能由本地测试代替。新源M/xs或任一回归失败就停止，不放宽门槛。

## 阶段1验证

本地 `python -B -m unittest tests.test_upper_snapshot tests.test_source_mouth -v`：25项通过。新增覆盖portrait准入/超尺寸拒绝、source-cache文件与数组hash、错源/错代码/越界路径拒绝、只返回三项源张量不混driver、真实supervisor状态和512MiB接口、原驱动template/自动crop钩子保留、新源只准student等。导入防火墙仍不加载Torch/CV2。

本阶段只完成本地实现及素材metadata；没有远程登录/写入、GPU运行、训练、图片生成、commit或push。真实完整四视频、两个对比、语义/闭口/牙形审阅均未执行，父审核发布后才进入运行阶段。
