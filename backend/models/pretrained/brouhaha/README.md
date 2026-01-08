# Brouhaha SNR + C50 检测模型

**版本**: V3.1.1+dev.20260107.03

## 模型信息

| 属性 | 值 |
|------|-----|
| **模型来源** | [pyannote/brouhaha](https://huggingface.co/pyannote/brouhaha) |
| **框架** | PyTorch (pyannote.audio) |
| **输入** | 16kHz 单声道音频 |
| **输出** | SNR (dB), C50 (dB), VAD (0-1) |
| **推理速度** | ~5ms/chunk (CPU), ~2ms/chunk (GPU) |
| **显存占用** | ~200MB (GPU) |

## 输出说明

| 输出 | 含义 | 典型范围 |
|------|------|---------|
| **SNR** | 信噪比 (Signal-to-Noise Ratio) | -10dB ~ 50dB |
| **C50** | 清晰度指数 (Clarity Index) | -20dB ~ 30dB |
| **VAD** | 语音活动检测 (Voice Activity Detection) | 0.0 ~ 1.0 |

## 获取模型

### 方式1：自动下载（推荐）

首次运行时，BrouhahaService 会自动从 HuggingFace 下载模型。

**前提条件**：
1. 设置 HuggingFace Token 环境变量
2. 访问 https://huggingface.co/pyannote/brouhaha 接受用户协议

```bash
# Windows PowerShell
$env:HUGGING_FACE_HUB_TOKEN="your_token_here"

# Windows CMD
set HUGGING_FACE_HUB_TOKEN=your_token_here

# Linux/Mac
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

### 方式2：手动下载

```bash
# 使用 huggingface-cli 下载
pip install huggingface-hub

huggingface-cli download pyannote/brouhaha \
    --local-dir backend/models/pretrained/brouhaha \
    --token your_hf_token_here
```

## 文件结构

下载完成后，目录应包含：

```
backend/models/pretrained/brouhaha/
├── pytorch_model.bin       # PyTorch 模型权重
├── config.yaml             # pyannote 模型配置
└── README.md               # 本说明文件
```

## 使用示例

```python
from app.services.brouhaha_service import get_brouhaha_service

# 获取服务单例
service = get_brouhaha_service()

# 检测音频
result = service.detect(audio, sr=16000)

print(f"SNR: {result.snr:.1f}dB")   # 信噪比
print(f"C50: {result.c50:.1f}dB")   # 清晰度指数
print(f"VAD: {result.vad:.2f}")     # 语音活动概率
```

## 回退机制

如果模型不可用，服务会自动回退到 WADA-SNR 算法：
- WADA-SNR 是一种无模型的 SNR 估计算法
- 不依赖深度学习，仅使用信号处理
- C50 将返回 0.0（无法估计）

## 参考文献

- [Brouhaha 论文](https://arxiv.org/abs/2210.13248)
- [pyannote.audio 文档](https://github.com/pyannote/pyannote-audio)
- [WADA-SNR 算法](https://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf)
