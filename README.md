# Qwen3-ASR

<br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/logo.png" width="400"/>
<p>

<p align="center">
&nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-asr">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen/Qwen3-ASR">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwen.ai/blog?id=qwen3asr">博客</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2601.21337">论文</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen3-ASR">Hugging Face 演示</a>&nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://modelscope.cn/studios/Qwen/Qwen3-ASR">ModelScope 演示</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">微信</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/qwen-speech-recognition">API</a>

</p>

我们发布了 **Qwen3-ASR** 系列，包括两款功能全面的语音识别模型，支持 52 种语言和方言的语言识别与自动语音识别（ASR）；此外，还包含一款全新的非自回归语音强制对齐模型，支持 11 种语言的文本与语音对齐。


## 最新动态
* 2026.6.26: 🤗🤗🤗 现已支持原生 Transformers！同时支持 `torch.compile`，可加速推理。使用示例请参阅新的模型卡：
  - [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
  - [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)
  - [Qwen/Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
* 2026.1.29: 🎉🎉🎉 我们发布了 [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) 系列（0.6B/1.7B）和 Qwen3-ForcedAligner-0.6B 模型。详情请参阅我们的[博客](https://qwen.ai/blog?id=qwen3asr)！


## 目录 <!-- omit in toc -->

- [概述](#概述)
  - [简介](#简介)
  - [模型架构](#模型架构)
  - [已发布模型介绍与下载](#已发布模型介绍与下载)
- [快速开始](#快速开始)
  - [环境配置](#环境配置)
  - [Python 包使用方法](#python-包使用方法)
    - [快速推理](#快速推理)
    - [vLLM 后端](#vllm-后端)
    - [流式推理](#流式推理)
    - [ForcedAligner 使用方法](#forcedaligner-使用方法)
  - [DashScope API 使用方法](#dashscope-api-使用方法)
- [启动本地网页界面演示](#启动本地网页界面演示)
  - [Gradio 演示](#gradio-演示)
  - [流式演示](#流式演示)
- [使用 vLLM 部署](#使用-vllm-部署)
- [微调](#微调)
- [Docker](#docker)
- [评测](#评测)
- [引用](#引用)


## 概述

### 简介

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/qwen3_asr_introduction.png" width="90%"/>
<p>

Qwen3-ASR 系列包含 Qwen3-ASR-1.7B 和 Qwen3-ASR-0.6B，支持 52 种语言和方言的语言识别与自动语音识别。两款模型均受益于大规模语音训练数据，以及基础模型 Qwen3-Omni 强大的音频理解能力。实验表明，1.7B 版本在开源 ASR 模型中达到领先水平，并可与最强的闭源商业 API 竞争。主要特点如下：

* **功能全面**：Qwen3-ASR-1.7B 和 Qwen3-ASR-0.6B 支持 30 种语言和 22 种中文方言的语言识别与语音识别，同时支持多个国家和地区的英语口音。

* **准确高效**：Qwen3-ASR 系列在复杂声学环境和具有挑战性的文本模式下，仍能保持高质量、稳健的识别效果。Qwen3-ASR-1.7B 在开源及内部评测基准上均表现出色。0.6B 版本兼顾准确率与效率，在并发数为 128 时，吞吐量可达实时速率的 2000 倍。两款模型均可通过单一模型统一支持流式与离线推理，并支持长音频转写。

* **创新且强大的强制对齐方案**：我们推出 Qwen3-ForcedAligner-0.6B，支持 11 种语言，可为最长 5 分钟语音中的任意粒度单元预测时间戳。评测表明，其时间戳精度优于基于端到端（E2E）的强制对齐模型。

* **完善的推理工具包**：除开源 Qwen3-ASR 系列的模型架构和权重外，我们还提供功能完善的推理框架，支持基于 vLLM 的批量推理、异步服务、流式推理、时间戳预测等功能。

### 模型架构

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/overview.jpg" width="100%"/>
<p>


### 已发布模型介绍与下载

以下为 Qwen3-ASR 模型的介绍与下载信息，请根据需求选择并下载适合的模型。

| 模型 | 支持的语言 | 支持的方言 | 推理模式 | 音频类型 |
|---|---|---|---|---|
| Qwen3-ASR-1.7B & Qwen3-ASR-0.6B | 中文 (zh), 英语 (en), 粤语 (yue), 阿拉伯语 (ar), 德语 (de), 法语 (fr), 西班牙语 (es), 葡萄牙语 (pt), 印度尼西亚语 (id), 意大利语 (it), 韩语 (ko), 俄语 (ru), 泰语 (th), 越南语 (vi), 日语 (ja), 土耳其语 (tr), 印地语 (hi), 马来语 (ms), 荷兰语 (nl), 瑞典语 (sv), 丹麦语 (da), 芬兰语 (fi), 波兰语 (pl), 捷克语 (cs), 菲律宾语 (fil), 波斯语 (fa), 希腊语 (el), 匈牙利语 (hu), 马其顿语 (mk), 罗马尼亚语 (ro) | 安徽, 东北, 福建, 甘肃, 贵州, 河北, 河南, 湖北, 湖南, 江西, 宁夏, 山东, 陕西, 山西, 四川, 天津, 云南, 浙江, 粤语 (香港口音), 粤语 (广东口音), 吴语, 闽南语. | 离线 / 流式 | 语音、清唱、带背景音乐的歌曲 |
| Qwen3-ForcedAligner-0.6B | 中文, 英语, 粤语, 法语, 德语, 意大利语, 日语, 韩语, 葡萄牙语, 俄语, 西班牙语 | -- | 非自回归（NAR） | 语音 |

通过 `qwen-asr` 包或 vLLM 加载模型时，会根据模型名称自动下载权重。如果运行环境不允许在执行过程中下载权重，可使用以下命令，将模型权重手动下载到本地目录：

```bash
# 通过 ModelScope 下载（推荐中国大陆用户使用）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B
# 通过 Hugging Face 下载
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./Qwen3-ForcedAligner-0.6B
```


## 快速开始

### 环境配置

使用 Qwen3-ASR 最简单的方式是从 PyPI 安装 `qwen-asr` Python 包。安装时会自动获取所需的运行依赖，随后即可加载任意已发布的 Qwen3-ASR 模型。若希望进一步简化环境配置，也可使用官方 [Docker 镜像](#docker)。`qwen-asr` 包提供 transformers 和 vLLM 两种后端，具体用法请参阅 [Python 包使用方法](#python-包使用方法)。建议使用**全新、独立的环境**，避免与已有包发生依赖冲突。可按以下方式创建一个干净的 Python 3.12 环境：

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
```

运行以下命令，完成支持 transformers 后端的最小安装：

```bash
pip install -U qwen-asr
```

如需启用 vLLM 后端，以获得更快的推理速度和流式支持，请运行：

```bash
pip install -U qwen-asr[vllm]
```

如需在本地开发或修改代码，请从源码以可编辑模式安装：

```bash
git clone https://github.com/QwenLM/Qwen3-ASR.git
cd Qwen3-ASR
pip install -e .
# 支持 vLLM 后端
# pip install -e ".[vllm]"
```

此外，建议使用 FlashAttention 2，以降低 GPU 显存占用并加速推理，尤其适用于长输入和较大的批量大小。

```bash
pip install -U flash-attn --no-build-isolation
```

如果机器内存小于 96GB，且 CPU 核心较多，请运行：

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

硬件也需要兼容 FlashAttention 2，详情请参阅 [FlashAttention 仓库](https://github.com/Dao-AILab/flash-attention)的官方文档。只有在模型以 `torch.float16` 或 `torch.bfloat16` 加载时，才能使用 FlashAttention 2。

### Python 包使用方法

#### 快速推理

`qwen-asr` 包提供两种后端：**transformers 后端**和 **vLLM 后端**。音频输入可使用本地路径、URL、base64 数据或 `(np.ndarray, sr)` 元组，并支持批量推理。如需快速体验 Qwen3-ASR，可通过 `Qwen3ASRModel.from_pretrained(...)` 使用 transformers 后端，示例如下：

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # 推理批量大小上限。-1 表示不限制；较小的值有助于避免显存不足（OOM）。
    max_new_tokens=256, # 生成 token 数量的上限。长音频输入可设置更大的值。
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    language=None, # 设为 "English" 可强制指定语言
)

print(results[0].language)
print(results[0].text)
```

如需返回时间戳，请传入 `forced_aligner` 及其初始化参数。以下示例展示了输出时间戳的批量推理：

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # 推理批量大小上限。-1 表示不限制；较小的值有助于避免显存不足（OOM）。
    max_new_tokens=256, # 生成 token 数量的上限。长音频输入可设置更大的值。
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
        # attn_implementation="flash_attention_2",
    ),
)

results = model.transcribe(
    audio=[
      "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
      "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    ],
    language=["Chinese", "English"], # 也可设为 None，以自动检测语言
    return_time_stamps=True,
)

for r in results:
    print(r.language, r.text, r.time_stamps[0])
```

更多用法请参阅 transformers 后端的[示例代码](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_transformers.py)。

#### vLLM 后端

如需获得最快的 Qwen3-ASR 推理速度，强烈建议使用 vLLM 后端，通过 `Qwen3ASRModel.LLM(...)` 初始化模型，示例如下。请先运行 `pip install -U qwen-asr[vllm]` 完成安装。如果需要输出时间戳，建议通过 `pip install -U flash-attn --no-build-isolation` 安装 FlashAttention，以加速强制对齐模型的推理。请将代码放在 `if __name__ == '__main__':` 保护块下，避免出现 [vLLM 故障排查](https://docs.vllm.ai/en/latest/usage/troubleshooting/#python-multiprocessing)中描述的 `spawn` 错误。

```python
import torch
from qwen_asr import Qwen3ASRModel

if __name__ == '__main__':
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.7,
        max_inference_batch_size=128, # 推理批量大小上限。-1 表示不限制；较小的值有助于避免显存不足（OOM）。
        max_new_tokens=4096, # 生成 token 数量的上限。长音频输入可设置更大的值。
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
        ),
    )

    results = model.transcribe(
        audio=[
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
        ],
        language=["Chinese", "English"], # 也可设为 None，以自动检测语言
        return_time_stamps=True,
    )

    for r in results:
        print(r.language, r.text, r.time_stamps[0])
```

更多用法请参阅 vLLM 后端的[示例代码](examples/example_qwen3_asr_vllm.py)。此外，也可通过 `qwen-asr-serve` 命令启动 vLLM 服务。该命令封装了 `vllm serve`，可传入其支持的任意参数，例如：

```bash
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
```

随后可通过以下方式向服务发送请求：

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
                    },
                }
            ],
        }
    ]
}

response = requests.post(url, headers=headers, json=data, timeout=300)
response.raise_for_status()
content = response.json()['choices'][0]['message']['content']
print(content)

# 按需解析 ASR 输出
from qwen_asr import parse_asr_output
language, text = parse_asr_output(content)
print(language)
print(text)
```

#### 流式推理

Qwen3-ASR 完整支持流式推理。目前，流式推理仅适用于 vLLM 后端，且不支持批量推理或返回时间戳。详情请参阅[示例代码](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py)。也可按照[指南](#流式演示)启动流式网页演示，体验 Qwen3-ASR 的流式转写能力。

#### ForcedAligner 使用方法

`Qwen3-ForcedAligner-0.6B` 可对齐文本与语音，并返回词级或字级时间戳。直接使用强制对齐模型的示例如下：

```python
import torch
from qwen_asr import Qwen3ForcedAligner

model = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
)

results = model.align(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    text="甚至出现交易几乎停滞的情况。",
    language="Chinese",
)

print(results[0])
print(results[0][0].text, results[0][0].start_time, results[0][0].end_time)
```

此外，强制对齐模型支持本地路径、URL、base64 数据、`(np.ndarray, sr)` 输入以及批量推理。详情请参阅[示例代码](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_forced_aligner.py)。

### DashScope API 使用方法

如需进一步体验 Qwen3-ASR，也可尝试 DashScope API，获得更快捷、高效的使用体验。API 详情及文档如下：

| API 说明 | API 文档（中国大陆） | API 文档（国际） |
|------------------|-----------------------------------|------------------------------------|
| Qwen3-ASR 实时语音识别 API。 | [https://help.aliyun.com/zh/model-studio/qwen-real-time-speech-recognition](https://help.aliyun.com/zh/model-studio/qwen-real-time-speech-recognition) | [https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition](https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition) |
| Qwen3-ASR 文件转写 API。 | [https://help.aliyun.com/zh/model-studio/qwen-speech-recognition](https://help.aliyun.com/zh/model-studio/qwen-speech-recognition) | [https://www.alibabacloud.com/help/en/model-studio/qwen-speech-recognition](https://www.alibabacloud.com/help/en/model-studio/qwen-speech-recognition) |


## 启动本地网页界面演示

### Gradio 演示

如需启动 Qwen3-ASR 的 Gradio 网页界面演示，请先安装 `qwen-asr` 包，然后运行 `qwen-asr-demo`。可通过以下命令查看帮助：

```bash
qwen-asr-demo --help
```

可使用以下命令启动演示：

```bash
# Transformers 后端
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --ip 0.0.0.0 --port 8000

# Transformers 后端 + 强制对齐模型（启用时间戳）
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"device_map":"cuda:0","dtype":"bfloat16","max_inference_batch_size":8,"max_new_tokens":256}' \
  --aligner-kwargs '{"device_map":"cuda:0","dtype":"bfloat16"}' \
  --ip 0.0.0.0 --port 8000

# vLLM 后端 + 强制对齐模型（启用时间戳）
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B \
  --backend vllm \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"gpu_memory_utilization":0.7,"max_inference_batch_size":8,"max_new_tokens":2048}' \
  --aligner-kwargs '{"device_map":"cuda:0","dtype":"bfloat16"}' \
  --ip 0.0.0.0 --port 8000
```

随后打开 `http://<your-ip>:8000`，或通过 VS Code 等工具的端口转发功能访问。

#### 后端说明

该演示支持 transformers 和 vLLM 两种后端。后端专用的初始化参数应以 JSON 字典形式通过 `--backend-kwargs` 传入。如果未提供，将使用合理的默认值。

```bash
# 示例：覆盖 transformers 初始化参数，启用 FlashAttention
--backend-kwargs '{"device_map":"cuda:0","dtype":"bfloat16","attn_implementation":"flash_attention_2"}'

# 示例：覆盖 vLLM 初始化参数，使用 65% 的 GPU 显存
--backend-kwargs '{"gpu_memory_utilization":0.65}'
```

#### CUDA 设备说明

由于 vLLM 不采用 `cuda:0` 形式的设备选择方式，该演示通过 `--cuda-visible-devices` 设置 `CUDA_VISIBLE_DEVICES` 来选择 GPU。

```bash
# 使用 GPU 0
--cuda-visible-devices 0

# 使用 GPU 1
--cuda-visible-devices 1
```

#### 时间戳说明

只有提供 `--aligner-checkpoint` 时，才支持时间戳。如果启动演示时未加载强制对齐模型，时间戳界面会自动隐藏。

```bash
# 不加载强制对齐模型
qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B

# 加载强制对齐模型
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B
```

#### HTTPS 说明

为避免服务部署后出现浏览器麦克风权限问题，建议通过 HTTPS 运行 Gradio 服务；在远程访问或使用现代浏览器、网关时，可能必须启用 HTTPS。可使用 `--ssl-certfile` 和 `--ssl-keyfile` 启用 HTTPS。首先生成私钥和自签名证书（有效期为 365 天）：

```bash
openssl req -x509 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=localhost"
```

随后以 HTTPS 启动演示：

```bash
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --ip 0.0.0.0 --port 8000 \
  --ssl-certfile cert.pem \
  --ssl-keyfile key.pem \
  --no-ssl-verify
```

打开 `https://<your-ip>:8000` 即可使用。如果浏览器显示警告，这是自签名证书的正常现象。生产环境请使用正式证书。

### 流式演示

为便于在网页界面中体验 Qwen3-ASR 的流式转写能力，我们提供了一个基于 Flask 的简易流式演示。该演示在浏览器中采集麦克风音频，将其重采样至 16,000 Hz，并持续向模型推送 PCM 音频块。可通过以下命令启动：

```bash
qwen-asr-demo-streaming \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8000
```

随后打开 `http://<your-ip>:8000`，或通过 VS Code 等工具的端口转发功能访问。

浏览器通过局域网 IP 采集麦克风音频时需要 HTTPS。流式演示同样支持
`--ssl-certfile` 和 `--ssl-keyfile`，例如在上述命令中加入：

```bash
  --max-model-len 8192 \
  --ssl-certfile cert.pem \
  --ssl-keyfile key.pem
```

使用 IP 地址访问时，证书的 SAN 应包含实际服务器 IP。下面以 `192.168.100.230` 为例：

```bash
openssl req -x509 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=192.168.100.230" \
  -addext "subjectAltName=IP:192.168.100.230,DNS:localhost,IP:127.0.0.1"
```

然后通过 `https://<your-ip>:<host-port>` 打开页面，并信任自签名证书、允许麦克风权限。
例如 Docker 映射为宿主机 `8001` 到容器 `80` 时，服务使用 `--port 80`，
浏览器访问 `https://192.168.100.230:8001/`。API 客户端也应切换到 HTTPS，
并配置该证书作为信任证书。

流式 demo 使用单句话的识别状态，原始实现会不断累积音频与文本前缀。
`--chunk-size-sec` 仅控制推理频率，不是上下文保留时长。长时间录音应自动分段：
本 demo 默认使用 `--segment-max-sec 30`，优先在单段超过一半时长后的低能量
停顿处结束分段，持续讲话则在最大时长处切段。已完成分段的文本保留在返回的
`text` 中，新的分段重新建立识别状态，避免历史音频持续占用模型上下文。
这是按音频停顿和时长分段，不是语义断句，也不识别说话人。
API 额外返回 `segments_completed`、`segment_seconds` 和 `total_audio_seconds`。
如配置了非常小的上下文，单个推理块仍可能无法容纳，此时返回 HTTP 422，
需要减小 `--chunk-size-sec`。持续讲话在强制切段处可能影响词语的连贯识别。

## 使用 vLLM 部署

vLLM 官方在 Qwen3-ASR 发布首日即提供了模型支持，可用于高效推理。

### 安装
可通过 vLLM 的 nightly wheel 包或 Docker 镜像运行 Qwen3-ASR。安装 vLLM 的 nightly 版本时，建议使用 `uv` 管理环境：
```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]" # 安装额外的音频依赖
```

### 在线服务
运行以下命令，即可通过 vLLM 部署 Qwen3-ASR：
```bash
vllm serve Qwen/Qwen3-ASR-1.7B
```
模型服务部署成功后，可通过多种方式与其交互。

#### 使用 OpenAI SDK
```python
import base64
import httpx
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# 创建多模态聊天补全请求
response = client.chat.completions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}
                    }
                }
            ]
        }
    ],
)

print(response.choices[0].message.content)
```
vLLM 也支持通过 OpenAI 转写 API 调用该模型。
```python
import httpx
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
audio_file = httpx.get(audio_url).content

transcription = client.audio.transcriptions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    file=audio_file,
)

print(transcription.text)
```

#### 使用 cURL
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "messages": [
    {"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}
    ]}
    ]
    }'
```

### 离线推理
以下示例展示如何通过 vLLM 对 Qwen3-ASR 进行离线推理：
```python
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
import base64
import requests

# 初始化 LLM
llm = LLM(
    model="Qwen/Qwen3-ASR-1.7B"
)

# 加载音频
audio_asset = AudioAsset("winning_call")

# 创建包含音频内容的对话
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": audio_asset.url}
            }
        ]
    }
]

sampling_params = SamplingParams(temperature=0.01, max_tokens=256)

# 使用 .chat() 进行推理
outputs = llm.chat(conversation, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```


## 微调

Qwen3-ASR 微调的详细说明请参阅 [Qwen3-ASR-Finetuning](finetuning/)。


## Docker

为方便使用 `qwen-asr` Python 包，我们提供了预构建的 Docker 镜像：[qwenllm/qwen3-asr](https://hub.docker.com/r/qwenllm/qwen3-asr)。只需安装 GPU 驱动并下载模型文件，即可运行代码。请按照 [NVIDIA Container Toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)配置环境，确保 Docker 能访问 GPU。如果在中国大陆访问 Docker Hub 遇到困难，可使用镜像加速服务加快镜像拉取。

首先拉取镜像并启动容器：

```bash
LOCAL_WORKDIR=/data/metahuman_work/qwen3-asr
HOST_PORT=8001
CONTAINER_PORT=80
docker run --gpus all --name qwen3-asr \
    -v /var/run/docker.sock:/var/run/docker.sock -p $HOST_PORT:$CONTAINER_PORT \
    --mount type=bind,source=$LOCAL_WORKDIR,target=/data/shared/Qwen3-ASR \
    --shm-size=4gb \
    -it qwenllm/qwen3-asr:latest
```

执行命令后，将进入容器的 Bash shell。本地工作目录（**请将** `/path/to/your/workspace` **替换为实际路径**）会挂载到容器内的 `/data/shared/Qwen3-ASR`。宿主机的 `8000` 端口映射到容器内的 `80` 端口，因此可通过 `http://<host-ip>:8000` 访问容器内运行的服务。注意，容器内的服务必须监听 `0.0.0.0`（而非 `127.0.0.1`），端口转发才能正常工作。

退出容器后，可通过以下命令重新启动并进入容器：

```bash
docker start qwen3-asr
docker exec -it qwen3-asr bash
```

如需彻底删除容器，请运行：

```bash
docker rm -f qwen3-asr
```


## 评测

评测时，我们使用 vLLM 对所有模型进行推理，设置 `dtype=torch.bfloat16` 和 `max_new_tokens=1024`。所有解码均采用贪心搜索，所有测试均未指定语言参数。详细评测结果如下。

<details>
<summary>公开数据集 ASR 评测（词错误率 WER ↓）</summary>

<table>
  <thead>
    <tr>
      <th colspan="2" style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">英语（en）</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Librispeech<br>clean | other</td>
      <td style="text-align: center;"><strong>1.39</strong> | 3.75</td>
      <td style="text-align: center;">2.89 | 3.56</td>
      <td style="text-align: center;">2.78 | 5.70</td>
      <td style="text-align: center;">1.51 | 3.97</td>
      <td style="text-align: center;">1.68 | 4.03</td>
      <td style="text-align: center;">2.11 | 4.55</td>
      <td style="text-align: center;">1.63 | <strong>3.38</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">GigaSpeech</td>
      <td style="text-align: center;">25.50</td>
      <td style="text-align: center;">9.37</td>
      <td style="text-align: center;">9.55</td>
      <td style="text-align: center;">9.76</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">8.88</td>
      <td style="text-align: center;"><strong>8.45</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-en</td>
      <td style="text-align: center;">9.08</td>
      <td style="text-align: center;">14.49</td>
      <td style="text-align: center;">13.78</td>
      <td style="text-align: center;">9.90</td>
      <td style="text-align: center;">9.90</td>
      <td style="text-align: center;">9.92</td>
      <td style="text-align: center;"><strong>7.39</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-en</td>
      <td style="text-align: center;"><strong>2.40</strong></td>
      <td style="text-align: center;">2.94</td>
      <td style="text-align: center;">6.31</td>
      <td style="text-align: center;">4.08</td>
      <td style="text-align: center;">5.49</td>
      <td style="text-align: center;">4.39</td>
      <td style="text-align: center;">3.35</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">MLS-en</td>
      <td style="text-align: center;">5.12</td>
      <td style="text-align: center;"><strong>3.68</strong></td>
      <td style="text-align: center;">7.09</td>
      <td style="text-align: center;">4.87</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">6.00</td>
      <td style="text-align: center;">4.58</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Tedlium</td>
      <td style="text-align: center;">7.69</td>
      <td style="text-align: center;">6.15</td>
      <td style="text-align: center;">4.91</td>
      <td style="text-align: center;">6.84</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>3.85<strong></td>
      <td style="text-align: center;"><strong>4.50</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">VoxPopuli</td>
      <td style="text-align: center;">10.29</td>
      <td style="text-align: center;">11.36</td>
      <td style="text-align: center;">12.12</td>
      <td style="text-align: center;">12.05</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>9.96<strong></td>
      <td style="text-align: center;"><strong>9.15</strong></td>
    </tr>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">中文（zh）</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech<br>net | meeting</td>
      <td style="text-align: center;">15.30 | 32.27</td>
      <td style="text-align: center;">14.43 | 13.47</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">9.86 | 19.11</td>
      <td style="text-align: center;">6.35 | -</td>
      <td style="text-align: center;">5.97 | 6.88</td>
      <td style="text-align: center;"><strong>4.97</strong> | <strong>5.88</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">AISHELL-2-test</td>
      <td style="text-align: center;">4.24</td>
      <td style="text-align: center;">11.62</td>
      <td style="text-align: center;">2.85</td>
      <td style="text-align: center;">5.06</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">3.15</td>
      <td style="text-align: center;"><strong>2.71</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">SpeechIO</td>
      <td style="text-align: center;">12.86</td>
      <td style="text-align: center;">5.30</td>
      <td style="text-align: center;">2.93</td>
      <td style="text-align: center;">7.56</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">3.44</td>
      <td style="text-align: center;"><strong>2.88</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-zh</td>
      <td style="text-align: center;">2.44</td>
      <td style="text-align: center;">2.71</td>
      <td style="text-align: center;">2.69</td>
      <td style="text-align: center;">4.09</td>
      <td style="text-align: center;">3.51</td>
      <td style="text-align: center;">2.88</td>
      <td style="text-align: center;"><strong>2.41</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-zh</td>
      <td style="text-align: center;">6.32</td>
      <td style="text-align: center;">7.70</td>
      <td style="text-align: center;">5.95</td>
      <td style="text-align: center;">12.91</td>
      <td style="text-align: center;">6.20</td>
      <td style="text-align: center;">6.89</td>
      <td style="text-align: center;"><strong>5.35</strong></td>
    </tr>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">中文方言</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">KeSpeech</td>
      <td style="text-align: center;">26.87</td>
      <td style="text-align: center;">24.71</td>
      <td style="text-align: center;">5.27</td>
      <td style="text-align: center;">28.79</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">7.08</td>
      <td style="text-align: center;"><strong>5.10</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-yue</td>
      <td style="text-align: center;">4.98</td>
      <td style="text-align: center;">9.43</td>
      <td style="text-align: center;">4.98</td>
      <td style="text-align: center;">9.18</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">5.79</td>
      <td style="text-align: center;"><strong>3.98</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-yue</td>
      <td style="text-align: center;">11.36</td>
      <td style="text-align: center;">18.76</td>
      <td style="text-align: center;">13.20</td>
      <td style="text-align: center;">16.23</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">9.50</td>
      <td style="text-align: center;"><strong>7.57</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-zh-tw</td>
      <td style="text-align: center;">6.32</td>
      <td style="text-align: center;">7.31</td>
      <td style="text-align: center;">4.06</td>
      <td style="text-align: center;">7.84</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">5.59</td>
      <td style="text-align: center;"><strong>3.77</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech-Yue<br>short | long</td>
      <td style="text-align: center;">15.62 | 25.29</td>
      <td style="text-align: center;">25.19 | 11.23</td>
      <td style="text-align: center;">9.74 | 11.40</td>
      <td style="text-align: center;">32.26 | 46.64</td>
      <td style="text-align: center;">- | -</td>
      <td style="text-align: center;">7.54 | 9.92</td>
      <td style="text-align: center;"><strong>5.82</strong> | <strong>8.85</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech-Chuan<br>easy | hard</td>
      <td style="text-align: center;">34.81 | 53.98</td>
      <td style="text-align: center;">43.79 | 67.30</td>
      <td style="text-align: center;"><strong>11.40<strong> | <strong>20.20</strong></td>
      <td style="text-align: center;">14.35 | 26.80</td>
      <td style="text-align: center;">- | -</td>
      <td style="text-align: center;">13.92 | 24.45</td>
      <td style="text-align: center;">11.99 | 21.63</td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>内部数据集 ASR 评测（词错误率 WER ↓）</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">带口音英语</td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Accented English</td>
      <td style="text-align: center;">28.56</td>
      <td style="text-align: center;">23.85</td>
      <td style="text-align: center;">20.41</td>
      <td style="text-align: center;">21.30</td>
      <td style="text-align: center;">19.96</td>
      <td style="text-align: center;"><strong>16.62<strong></td>
      <td style="text-align: center;"><strong>16.07</strong></td>
    </tr>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">普通话</td>
    </tr>
    <tr>
      <td style="text-align: left;">Elders&Kids</td>
      <td style="text-align: center;">14.27</td>
      <td style="text-align: center;">36.93</td>
      <td style="text-align: center;">4.17</td>
      <td style="text-align: center;">10.61</td>
      <td style="text-align: center;">4.54</td>
      <td style="text-align: center;">4.48</td>
      <td style="text-align: center;"><strong>3.81</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">ExtremeNoise</td>
      <td style="text-align: center;">36.11</td>
      <td style="text-align: center;">29.06</td>
      <td style="text-align: center;">17.04</td>
      <td style="text-align: center;">63.17</td>
      <td style="text-align: center;">36.55</td>
      <td style="text-align: center;">17.88</td>
      <td style="text-align: center;"><strong>16.17</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">TongueTwister</td>
      <td style="text-align: center;">20.87</td>
      <td style="text-align: center;">4.97</td>
      <td style="text-align: center;">3.47</td>
      <td style="text-align: center;">16.63</td>
      <td style="text-align: center;">9.02</td>
      <td style="text-align: center;">4.06</td>
      <td style="text-align: center;"><strong>2.44</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Mandarin</td>
      <td style="text-align: center;">20.73</td>
      <td style="text-align: center;">12.50</td>
      <td style="text-align: center;">6.61</td>
      <td style="text-align: center;">14.01</td>
      <td style="text-align: center;">7.32</td>
      <td style="text-align: center;">7.06</td>
      <td style="text-align: center;"><strong>6.54</strong></td>
    </tr>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">中文方言</td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Cantonese</td>
      <td style="text-align: center;">16.05</td>
      <td style="text-align: center;">14.98</td>
      <td style="text-align: center;">7.56</td>
      <td style="text-align: center;">31.04</td>
      <td style="text-align: center;">5.85</td>
      <td style="text-align: center;"><strong>4.80<strong></td>
      <td style="text-align: center;"><strong>4.12</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Chinese Dialects</td>
      <td style="text-align: center;">45.37</td>
      <td style="text-align: center;">47.70</td>
      <td style="text-align: center;">19.85</td>
      <td style="text-align: center;">44.55</td>
      <td style="text-align: center;">19.41</td>
      <td style="text-align: center;"><strong>18.24<strong></td>
      <td style="text-align: center;"><strong>15.94</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>方言覆盖范围：</strong> <em>Dialog-Accented English</em> 的结果为 16 种口音的平均值，<em>Dialog-Chinese Dialects</em> 的结果为 22 种中文方言的平均值。</p>

</details>

<details>
<summary>多语言 ASR 评测（词错误率 WER ↓）</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GLM-ASR<br>-Nano-2512</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="6" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">开源评测基准</td>
    </tr>
    <tr>
      <td style="text-align: left;">MLS</td>
      <td style="text-align: center;">13.32</td>
      <td style="text-align: center;">8.62</td>
      <td style="text-align: center;">28.70</td>
      <td style="text-align: center;">13.19</td>
      <td style="text-align: center;"><strong>8.55</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">CommonVoice</td>
      <td style="text-align: center;">19.40</td>
      <td style="text-align: center;">10.77</td>
      <td style="text-align: center;">17.25</td>
      <td style="text-align: center;">12.75</td>
      <td style="text-align: center;"><strong>9.18</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MLC-SLM</td>
      <td style="text-align: center;">34.93</td>
      <td style="text-align: center;">15.68</td>
      <td style="text-align: center;">29.94</td>
      <td style="text-align: center;">15.84</td>
      <td style="text-align: center;"><strong>12.74</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs</td>
      <td style="text-align: center;">16.08</td>
      <td style="text-align: center;">5.27</td>
      <td style="text-align: center;">10.03</td>
      <td style="text-align: center;">7.57</td>
      <td style="text-align: center;"><strong>4.90</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs<sup>†</sup></td>
      <td style="text-align: center;">20.05</td>
      <td style="text-align: center;">6.85</td>
      <td style="text-align: center;">31.89</td>
      <td style="text-align: center;">10.37</td>
      <td style="text-align: center;"><strong>6.62</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs<sup>††</sup></td>
      <td style="text-align: center;">24.83</td>
      <td style="text-align: center;"><strong>8.16</strong></td>
      <td style="text-align: center;">47.84</td>
      <td style="text-align: center;">21.80</td>
      <td style="text-align: center;">12.60</td>
    </tr>
    <tr>
      <td colspan="6" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Qwen-ASR 内部评测基准</td>
    </tr>
    <tr>
      <td style="text-align: left;">News-Multilingual</td>
      <td style="text-align: center;">49.40</td>
      <td style="text-align: center;">14.80</td>
      <td style="text-align: center;">65.07</td>
      <td style="text-align: center;">17.39</td>
      <td style="text-align: center;"><strong>12.80</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>语言覆盖范围：</strong> <em>MLS</em> 包含 8 种语言：{da, de, en, es, fr, it, pl, pt}.<br><em>CommonVoice</em> 包含 13 种语言：{en, zh, yue, zh_TW, ar, de, es, fr, it, ja, ko, pt, ru}.<br><em>MLC-SLM</em> 包含 11 种语言：{en, fr, de, it, pt, es, ja, ko, ru, th, vi}.<br><em>Fleurs</em> 包含 12 种语言：{en, zh, yue, ar, de, es, fr, it, ja, ko, pt, ru }.<br><em>Fleurs<sup>†</sup></em> 在 Fleurs 基础上新增 8 种语言：{hi, id, ms, nl, pl, th, tr, vi}.<br><em>Fleurs<sup>††</sup></em> 在 Fleurs<sup>†</sup> 基础上新增 10 种语言：{cs, da, el, fa, fi, fil, hu, mk, ro, sv}。<br><em>News-Multilingual</em> 包含 15 种语言：{ar, de, es, fr, hi, id, it, ja, ko, nl, pl, pt, ru, th, vi}.</p>

</details>

<details>
<summary>语言识别准确率（%）↑</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">Whisper-large-v3</th>
      <th style="text-align: center;">Qwen3-ASR-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">MLS</td>
      <td style="text-align: center;"><strong>99.9</strong></td>
      <td style="text-align: center;">99.3</td>
      <td style="text-align: center;"><strong>99.9</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">CommonVoice</td>
      <td style="text-align: center;">92.7</td>
      <td style="text-align: center;"><strong>98.2<strong></td>
      <td style="text-align: center;"><strong>98.7</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MLC-SLM</td>
      <td style="text-align: center;">89.2</td>
      <td style="text-align: center;"><strong>92.7<strong></td>
      <td style="text-align: center;"><strong>94.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs</td>
      <td style="text-align: center;">94.6</td>
      <td style="text-align: center;"><strong>97.1<strong></td>
      <td style="text-align: center;"><strong>98.7</strong></td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td style="text-align: left;"><em>平均</em></td>
      <td style="text-align: center;">94.1</td>
      <td style="text-align: center;"><strong>96.8<strong></td>
      <td style="text-align: center;"><strong>97.9</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>语言覆盖范围：</strong> 语言集合与多语言 ASR 评测相同。此处的 Fleurs 对应多语言 ASR 评测中的 Fleurs<sup>††</sup>，覆盖 30 种语言。</p>

</details>

<details>
<summary>清唱与歌曲转写（词错误率 WER ↓）</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR<br>-1.0</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR-MLT<br>-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="7" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">清唱</td>
    </tr>
    <tr>
      <td style="text-align: left;">M4Singer</td>
      <td style="text-align: center;">16.77</td>
      <td style="text-align: center;">20.88</td>
      <td style="text-align: center;">7.88</td>
      <td style="text-align: center;">13.58</td>
      <td style="text-align: center;">7.29</td>
      <td style="text-align: center;"><strong>5.98</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MIR-1k-vocal</td>
      <td style="text-align: center;">11.87</td>
      <td style="text-align: center;">9.85</td>
      <td style="text-align: center;">6.56</td>
      <td style="text-align: center;">11.71</td>
      <td style="text-align: center;">8.17</td>
      <td style="text-align: center;"><strong>6.25</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Opencpop</td>
      <td style="text-align: center;">7.93</td>
      <td style="text-align: center;">6.49</td>
      <td style="text-align: center;">3.80</td>
      <td style="text-align: center;">9.52</td>
      <td style="text-align: center;"><strong>2.98</strong></td>
      <td style="text-align: center;">3.08</td>
    </tr>
    <tr>
      <td style="text-align: left;">Popcs</td>
      <td style="text-align: center;">32.84</td>
      <td style="text-align: center;">15.13</td>
      <td style="text-align: center;">8.97</td>
      <td style="text-align: center;">13.77</td>
      <td style="text-align: center;">9.42</td>
      <td style="text-align: center;"><strong>8.52</strong></td>
    </tr>
    <tr>
      <td colspan="7" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">带背景音乐的歌曲</td>
    </tr>
    <tr>
      <td style="text-align: left;">EntireSongs-en</td>
      <td style="text-align: center;">30.71</td>
      <td style="text-align: center;"><strong>12.18</strong></td>
      <td style="text-align: center;">33.51</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">14.60</td>
    </tr>
    <tr>
      <td style="text-align: left;">EntireSongs-zh</td>
      <td style="text-align: center;">34.86</td>
      <td style="text-align: center;">18.68</td>
      <td style="text-align: center;">23.99</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;"><strong>13.91</strong></td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>ASR 不同推理模式的表现（词错误率 WER ↓）</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;">模型</th>
      <th style="text-align: left;">推理模式</th>
      <th style="text-align: center;">Librispeech</th>
      <th style="text-align: center;">Fleurs-en</th>
      <th style="text-align: center;">Fleurs-zh</th>
      <th style="text-align: center;">平均</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="text-align: left; vertical-align: middle;">Qwen3-ASR-1.7B</td>
      <td style="text-align: left;">离线</td>
      <td style="text-align: center;">1.63 | 3.38</td>
      <td style="text-align: center;">3.35</td>
      <td style="text-align: center;">2.41</td>
      <td style="text-align: center;">2.69</td>
    </tr>
    <tr>
      <td style="text-align: left;">流式</td>
      <td style="text-align: center;">1.95 | 4.51</td>
      <td style="text-align: center;">4.02</td>
      <td style="text-align: center;">2.84</td>
      <td style="text-align: center;">3.33</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td rowspan="2" style="text-align: left; vertical-align: middle;">Qwen3-ASR-0.6B</td>
      <td style="text-align: left;">离线</td>
      <td style="text-align: center;">2.11 | 4.55</td>
      <td style="text-align: center;">4.39</td>
      <td style="text-align: center;">2.88</td>
      <td style="text-align: center;">3.48</td>
    </tr>
    <tr>
      <td style="text-align: left;">流式</td>
      <td style="text-align: center;">2.54 | 6.27</td>
      <td style="text-align: center;">5.38</td>
      <td style="text-align: center;">3.40</td>
      <td style="text-align: center;">4.40</td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>强制对齐评测（AAS，毫秒 ↓）</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">Monotonic-Aligner</th>
      <th style="text-align: center;">NFA</th>
      <th style="text-align: center;">WhisperX</th>
      <th style="text-align: center;">Qwen3-ForcedAligner-0.6B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">MFA 标注的原始音频</td>
    </tr>
    <tr>
      <td style="text-align: left;">中文</td>
      <td style="text-align: center;">161.1</td>
      <td style="text-align: center;">109.8</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>33.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">英语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">107.5</td>
      <td style="text-align: center;">92.1</td>
      <td style="text-align: center;"><strong>37.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">法语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">100.7</td>
      <td style="text-align: center;">145.3</td>
      <td style="text-align: center;"><strong>41.7</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">德语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">122.7</td>
      <td style="text-align: center;">165.1</td>
      <td style="text-align: center;"><strong>46.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">意大利语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">142.7</td>
      <td style="text-align: center;">155.5</td>
      <td style="text-align: center;"><strong>75.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">日语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">韩语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>37.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">葡萄牙语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>38.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">俄语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">200.7</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>40.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">西班牙语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">124.7</td>
      <td style="text-align: center;">108.0</td>
      <td style="text-align: center;"><strong>36.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>平均</em></td>
      <td style="text-align: center;">161.1</td>
      <td style="text-align: center;">129.8</td>
      <td style="text-align: center;">133.2</td>
      <td style="text-align: center;"><strong>42.9</strong></td>
    </tr>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">MFA 标注的拼接音频（300 秒）</td>
    </tr>
    <tr>
      <td style="text-align: left;">中文</td>
      <td style="text-align: center;">1742.4</td>
      <td style="text-align: center;">235.0</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>36.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">英语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">226.7</td>
      <td style="text-align: center;">227.2</td>
      <td style="text-align: center;"><strong>58.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">法语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">230.6</td>
      <td style="text-align: center;">2052.2</td>
      <td style="text-align: center;"><strong>53.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">德语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">220.3</td>
      <td style="text-align: center;">993.4</td>
      <td style="text-align: center;"><strong>62.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">意大利语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">290.5</td>
      <td style="text-align: center;">5719.4</td>
      <td style="text-align: center;"><strong>81.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">日语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>81.3</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">韩语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">葡萄牙语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>50.0</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">俄语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">283.3</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>43.0</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">西班牙语</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">240.2</td>
      <td style="text-align: center;">4549.9</td>
      <td style="text-align: center;"><strong>39.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">跨语言</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>34.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>平均</em></td>
      <td style="text-align: center;">1742.4</td>
      <td style="text-align: center;">246.7</td>
      <td style="text-align: center;">2708.4</td>
      <td style="text-align: center;"><strong>52.9</strong></td>
    </tr>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">人工标注</td>
    </tr>
    <tr>
      <td style="text-align: left;">原始音频</td>
      <td style="text-align: center;">49.9</td>
      <td style="text-align: center;">88.6</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>27.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">带噪原始音频</td>
      <td style="text-align: center;">53.3</td>
      <td style="text-align: center;">89.5</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>41.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">拼接音频（60 秒）</td>
      <td style="text-align: center;">51.1</td>
      <td style="text-align: center;">86.7</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>25.3</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">拼接音频（300 秒）</td>
      <td style="text-align: center;">410.8</td>
      <td style="text-align: center;">140.0</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>24.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">跨语言拼接音频</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>平均</em></td>
      <td style="text-align: center;">141.3</td>
      <td style="text-align: center;">101.2</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>32.4</strong></td>
    </tr>
  </tbody>
</table>

</details>


## 引用

如果我们的论文和代码对您的研究有帮助，欢迎点亮 Star :star: 并引用论文 :pencil: :)

```BibTeX
@article{Qwen3-ASR,
  title={Qwen3-ASR Technical Report},
  author={Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, Junyang Lin},
  journal={arXiv preprint arXiv:2601.21337},
  year={2026}
}
```


## Star 历史

[![Star 历史图表](https://api.star-history.com/svg?repos=QwenLM/Qwen3-ASR&type=Date)](https://star-history.com/#QwenLM/Qwen3-ASR&Date)

<br>
