# G 内部上牙适配器：仅连接性原型

本改动不是 RGB 视频后处理、牙形效果训练或优化版本。结构预测器、自动语义范围、上唇独立遮挡预测和跨帧上颌绑定均未实现；无视觉批准。不接入默认推理，不修改原权重。

## 最小通路

`SPADEDecoder.forward_features` 返回 `up_1` 后原生 64 通道特征；`decode_features` 保持原 leaky_relu → 3×3 conv → PixelShuffle2 → sigmoid 顺序。构造函数和 state_dict 键不变，默认 forward 仅串联两段。

`UpperTeethAdapterDecoder(base)` 包裹已加载 G；冻结其参数，wrapper.train() 后仍强制 base.eval()，防止 spectral norm buffer 更新。无条件调用直接 base(feature)。条件必须全部提供：

- structure `[B,3,Hfeat,Wfeat]`：规范牙冠占位、纵向坐标、浅分界条件，仅定义接口。
- visibility `[B,1,Hout,Wout]`：独立可见性软量。
- allowed bool `[B,1,Hout,Wout]`：显式输出许可区，不是自动牙分割。

浮点条件必须与输入 feature 的 dtype 一致、与 hidden 同设备，有限且在 [0,1]；autocast 内显式转为 hidden dtype。输出尺寸是 hidden 的两倍。不接受非指定尾部。

hidden 在 no_grad 中计算。64+3+1 通道经 1×1 Conv(32) → ReLU → 1×1 Conv(64)，末层零初始化。增量为 `max_delta*tanh(raw)*safe_mask*avgpool(visibility)`，默认上限 .1，最大 .5；加入 hidden 后以冻结但可微的原尾部解码。没有 parse_output、RGB 补片或强制区外替回。零初始化不按参数值绕过梯度。

支持域先交 `allowed & visibility>0`，要求每个 PixelShuffle 2×2 输出块全许可，再低分辨率 3×3 腐蚀（外部显式 False）。输出影响域是其 3×3 膨胀再重复 2×2，数学上属于许可区；真实 CUDA 数值仍必须实测区外严格零差，不以理论证明替代。窄区/空区/全零可见性无残留状态。

## 本地检查（不加载 Torch/CV2）

```text
python -B -m unittest discover -s tests -p test_upper_teeth_adapter_contract.py -v
python -B -m unittest discover -s tests -p test_upper_teeth_adapter_torch.py -v
```

第二条在 Windows 或未设置 `UPPER_ADAPTER_TENSOR_TESTS=1` 时，在任何 Torch/module import 前跳过。Linux 显式启用后使用小随机 G 测试，不取代真实 G 探针。NumPy/AST 检查支持域洞/边/空/窄、条件范围及拆分前后构造和数学语句一致。

## 父代理远程操作前门槛

先按工作区服务器规则读取服务器规范、重新检查 GPU UUID 和用户总额度。此脚本不会判断其他任务归属，不可代替资源授权。仅 Linux、现有隔离环境、单张显式可见 GPU；不安装、不下载。先发布审核后的独立 SHA，并使 Git 干净，设置 `PROBE_CODE_SHA` 为完整 HEAD。

所有源码/输入/模型/输出以及 HOME、TMPDIR、XDG_CACHE_HOME、TORCH_HOME、HF_HOME、CUDA_CACHE_PATH 必须解析到 workspace 内（缓存目录预先存在），设置 `PYTHONDONTWRITEBYTECODE=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，禁止输出落入受 Git 跟踪的工作树。

```text
python -B scripts/probe_upper_teeth_adapter.py --workspace <project-root> --feature-npz <causal-source_feature.npz> --keypoints-npz <baseline-f000263_finalK.npz> --provenance-json <original-causal-probe.json> --output <new-run-directory> --authorize-experimental
```

provenance 必须是原 causal probe 的 completed/postflight_verified 记录；使用 baseline.frames 中唯一 f263 的 f_s、prepare_source、x_s、final_k_hash，并核验全部五份原权重 SHA。NPZ 仅数值，不读外来 pickle。只实际加载 W/G 的已核验权重，使用 `weights_only=True`。

通过 `git show 31c26a497cedf336a813e18b61e96239f7cab878:src/modules/spade_generator.py` 在正确 package 下执行可信历史源码，同原权重严格加载，独立比较 state_dict 和真实输出，非新 G 自比。

探针检查 FP32/FP16 下历史 G、拆分 G、默认调用与 zero-init 的 tensor/uint8 精确一致；真实 W 特征上使用**合成矩形与合成条件**做 3 步 Adam，只更新 adapter，目标仅 baseline 加 .01 局部数值偏移，不是客户牙形 GT。验证有限非零梯度、实际变化、更新后FP32/FP16的float/uint8区外零差、空/不可见回退、全部 base 参数与 buffer 不变、五权重文件不变以及 adapter-only `synthetic-smoke.pt` 的 weights_only 复载精确一致。另以更新后的head在FP32/FP16检查1像素薄条、矩形内洞、贴画布边界的合成mask；零安全特征格必须精确回退。没有导出人脸图片或视频。

外层只监督自己新建的进程组，300 秒硬超时，GNU du 实际分配空间项目上限 20GiB、输出上限 128MiB；超限/差异/验证错误立即失败，不删文件、不宣成功。只写checkpoint和JSON，不导出人脸美化图或大量帧文件。probe-partial.json是验收前中间值，accepted=false，不是成功记录；失败目录保留供父代理检查。

## 尚未验证/不能推出

本地只允许 NumPy/AST，真实 Torch、autocast、梯度、CUDA 零差与复载均需父代理远程执行。即使探针成功也只证明 model-internal connectivity；不证明牙齿范围正确、自然度改善、全片稳定或 pasteback 区外保护。结构生成、训练数据准入、真实牙形训练和连续视觉验收仍缺失。合成检查点仅供机制复载测试，不用于正式推理或作为真实牙形训练的初始化；后者应重新零初始化。
