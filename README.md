# comfyui-ccip

这是一个轻量级的仓库，包含从原始 `imgutils` 项目中独立出来的 CCIP 推理模块 `ccip_lib`，用于对动漫人物进行特征提取、成对差异计算与聚类。

**主要内容**
- `ccip_lib/`：独立的 CCIP 推理包（可作为子模块或单独库使用）。
  - `ccip_lib/ccip.py`：核心 API（`ccip_extract_feature`, `ccip_batch_extract_features`, `ccip_difference`, `ccip_same`, `ccip_clustering`, `ccip_merge` 等）。
  - `ccip_lib/cli.py`：命令行入口，支持 `extract`、`diff`、`same` 命令并支持本地模型目录传入。
  - `ccip_lib/data/` 和 `ccip_lib/utils/`：为独立运行 vendored 的最小辅助模块（图片加载、ONNX 打开、缓存装饰器）。
  - `ccip_lib/README.md`：本地模型目录结构说明。
  - `requirements.txt`：运行所需的第三方依赖清单。

快速开始

1. 安装依赖（建议在虚拟环境中执行）：

```powershell
cd <仓库根目录>
python -m pip install -r ccip_lib/requirements.txt
```

2. 下载或准备本地模型目录，至少包含：
- `model_feat.onnx`
- `model_metrics.onnx`
- `metrics.json`

参阅 `ccip_lib/MODEL_STRUCTURE.md` 获取详细说明。

3. 运行命令行示例：

```powershell
cd ccip_lib
# 提取单张图片的特征并保存
python cli.py extract path\to\image.jpg -m C:\models\ccip-dir -o feat.npy

# 计算两张图像的差异
python cli.py diff img1.jpg img2.jpg -m C:\models\ccip-dir

# 判断两张图像是否为同一角色
python cli.py same img1.jpg img2.jpg -m C:\models\ccip-dir --threshold 0.18
```

说明与注意事项
- 若未传入 `--model-dir`，代码会尝试从 Hugging Face 仓库下载模型（需要网络）。
- 要做实际 ONNX 推理，需安装 `onnxruntime` 或 `onnxruntime-gpu`。
- `ccip_lib` 已尽量做到独立，但仍依赖常见第三方包（参见 `ccip_lib/requirements.txt`）。
