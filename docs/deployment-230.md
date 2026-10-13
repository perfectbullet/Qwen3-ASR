# Qwen3-ASR：230 部署、启动与测试说明

部署状态核对日期：2026-10-12（北京时间）。

## 1. 当前部署情况

当前运行的是 **Qwen3-ASR 流式 HTTP API 与麦克风演示页面**，通过 HTTPS 访问：

**https://192.168.100.230:8001/**

| 项目 | 当前配置 |
|---|---|
| SSH 登录 | `zenking@192.168.100.230` |
| GPU | NVIDIA GeForce RTX 4080，16GB |
| Docker 镜像 | `qwenllm/qwen3-asr:latest` |
| 容器名称 | `qwen3-asr` |
| 宿主机端口 → 容器端口 | `8001 → 80` |
| 宿主机挂载目录 | `/data/metahuman_work/qwen3-asr` |
| 容器内挂载目录 | `/data/shared/Qwen3-ASR` |
| 模型目录（容器内） | `/data/shared/Qwen3-ASR/Qwen3-ASR-1.7B` |
| 服务脚本（容器内） | `/data/shared/Qwen3-ASR/demo_streaming_with_max_len.py` |
| 服务日志（容器内） | `/data/shared/Qwen3-ASR/qwen-asr-streaming-https.log` |
| 模型上下文上限 | `8192` token，包含文本与音频编码后的 token |
| GPU 显存利用率参数 | `0.8` |
| 推理间隔 | `--chunk-size-sec 1.0` |
| 单段最长音频 | `--segment-max-sec 30` |
| HTTPS 服务 | Uvicorn + Flask WSGI 应用 |
| 自动启动 | 容器重启策略为 `no`；服务目前需手动启动 |

镜像内实际版本：`qwen-asr 0.0.4`、`vllm 0.14.0`、`transformers 4.57.6`、
`torch 2.9.1`、`Flask 3.1.2`、`uvicorn 0.40.0`。

**8001 目前提供 `/api/start`、`/api/chunk`、`/api/finish`。此前的
OpenAI `/v1/audio/transcriptions` 服务已停止，不能用整文件测试脚本直接测试当前端口。**

## 2. 脚本与文件位置

本地仓库：`/home/zj/Qwen3-ASR`。

| 本地文件 | 用途 |
|---|---|
| `qwen_asr/cli/demo_streaming.py` | 当前流式服务的源代码 |
| `examples/test_streaming_api.py` | HTTP 分块测试客户端 |
| `tests/test_streaming_sessions.py` | 不加载模型的回归测试 |
| `examples/test_openai_transcription.py` | 历史整文件 OpenAI API 测试客户端 |
| `asr_audio_dump_20260513_035431.wav` | 6.4 秒测试音频，16kHz、单声道 |

部署时，流式源代码被复制到宿主机：

```text
/data/metahuman_work/qwen3-asr/demo_streaming_with_max_len.py
```

Docker 挂载后，对应容器内的同名文件。当前启动命令使用这个文件。
镜像中原始 `qwen-asr-demo-streaming` 入口没有同步这些修改，因此下面新增的
参数应配合部署脚本使用。

当前挂载目录包含：

```text
/data/shared/Qwen3-ASR/
├── Qwen3-ASR-1.7B/
├── demo_streaming_with_max_len.py
├── demo_streaming_before_segmentation.py   # 自动分段修改前的备份
├── test_streaming_sessions.py
├── qwen-asr-streaming-https.log
└── tls/
    ├── cert.pem                           # HTTPS 公开证书
    ├── cert.cer                           # Windows 可导入的公开证书
    └── key.pem                            # HTTPS 私钥
```

## 3. 登录、查看状态与停止服务

### 3.1 登录服务器

在你的电脑上执行：

```bash
ssh zenking@192.168.100.230
```

### 3.2 检查容器和服务

在 **230 宿主机**执行：

```bash
docker ps -a --filter name=qwen3-asr
docker exec qwen3-asr ps -ef
docker exec qwen3-asr nvidia-smi
```

正常运行时，进程中会看到 `demo_streaming_with_max_len.py` 和 `VLLM::EngineCore`。
启动新实例前先确认现有实例已经停止，避免重复加载模型和争用端口。

### 3.3 停止或重启

停止服务及其容器，在 **230 宿主机**执行：

```bash
docker stop --time 30 qwen3-asr
```

重新启动容器：

```bash
docker start qwen3-asr
```

然后执行下一节的服务启动命令。单独启动容器不会自动启动 ASR 服务。
容器停止会结束当前录音会话及容器内的交互 Shell。
单独结束 Python 父进程可能留下占用显存的 vLLM 子进程，停止容器可以一并清理。

## 4. 启动流式 HTTPS 服务

### 4.1 在宿主机后台启动（当前实际命令）

在 **230 宿主机**执行；已有服务正在运行时无需重复执行：

```bash
docker exec -d qwen3-asr bash -lc '
exec python -u /data/shared/Qwen3-ASR/demo_streaming_with_max_len.py \
  --asr-model-path /data/shared/Qwen3-ASR/Qwen3-ASR-1.7B \
  --host 0.0.0.0 \
  --port 80 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --chunk-size-sec 1.0 \
  --segment-max-sec 30 \
  --ssl-certfile /data/shared/Qwen3-ASR/tls/cert.pem \
  --ssl-keyfile /data/shared/Qwen3-ASR/tls/key.pem \
  > /data/shared/Qwen3-ASR/qwen-asr-streaming-https.log 2>&1
'
```

`docker exec -d` 返回后模型仍可能在加载，以日志中的
`Uvicorn running on https://0.0.0.0:80` 为服务监听成功的标志。

查看日志，在 **230 宿主机**执行：

```bash
docker exec qwen3-asr tail -f /data/shared/Qwen3-ASR/qwen-asr-streaming-https.log
```

退出 `tail -f` 使用 `Ctrl+C`，不会停止后台 ASR 服务。

### 4.2 进入容器前台启动

先确认后台实例已停止。在 **230 宿主机**执行：

```bash
docker exec -it qwen3-asr bash
```

再在 **容器内**执行：

```bash
python -u /data/shared/Qwen3-ASR/demo_streaming_with_max_len.py \
  --asr-model-path /data/shared/Qwen3-ASR/Qwen3-ASR-1.7B \
  --host 0.0.0.0 \
  --port 80 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --chunk-size-sec 1.0 \
  --segment-max-sec 30 \
  --ssl-certfile /data/shared/Qwen3-ASR/tls/cert.pem \
  --ssl-keyfile /data/shared/Qwen3-ASR/tls/key.pem
```

前台模式便于直接观察日志；终止后检查 vLLM 子进程是否已经退出。

## 5. HTTPS 与网页麦克风测试

### 5.1 当前证书

证书包含 `192.168.100.230`、`localhost`、`127.0.0.1`，为自签名证书，
有效期至 2027-10-12。通过局域网 IP 使用麦克风需要安全上下文，
因此访问地址使用 `https://`。

现有证书已经生成，启动服务时直接使用即可。需要重新生成时，在 **容器内**执行：

```bash
mkdir -p /data/shared/Qwen3-ASR/tls
umask 077
openssl req -x509 -newkey rsa:2048 \
  -keyout /data/shared/Qwen3-ASR/tls/key.pem \
  -out /data/shared/Qwen3-ASR/tls/cert.pem \
  -days 365 -nodes \
  -subj "/CN=192.168.100.230" \
  -addext "subjectAltName=IP:192.168.100.230,DNS:localhost,IP:127.0.0.1"
chmod 644 /data/shared/Qwen3-ASR/tls/cert.pem
openssl x509 -in /data/shared/Qwen3-ASR/tls/cert.pem \
  -outform DER -out /data/shared/Qwen3-ASR/tls/cert.cer
chmod 644 /data/shared/Qwen3-ASR/tls/cert.cer
```

这组命令会替换现有证书和私钥，随后需要重启服务，客户端也需要更新信任证书。

### 5.2 网页测试

1. 打开 `https://192.168.100.230:8001/`。
2. 处理自签名证书提示并信任证书。
3. 点击 **Start / 开始**，允许浏览器使用麦克风。
4. 连续讲话，页面持续更新语言与累计识别文本。
5. 点击 **Stop / 停止**，服务处理尾音并释放会话。

浏览器每次发送约 500ms 音频，服务器每累计 1 秒进行一次增量识别。
刷新页面或重启服务后，需要重新开始录音。

### 5.3 Windows 导入公开证书（按需）

若提示 `NET::ERR_CERT_AUTHORITY_INVALID`，可以在 **打开网页的 Windows 电脑**
的 PowerShell 中执行：

```powershell
scp zenking@192.168.100.230:/data/metahuman_work/qwen3-asr/tls/cert.cer "$env:USERPROFILE\Downloads\qwen-asr-230.cer"
Import-Certificate -FilePath "$env:USERPROFILE\Downloads\qwen-asr-230.cer" -CertStoreLocation Cert:\CurrentUser\Root
```

确认导入后，关闭并重新打开浏览器，再访问 HTTPS 页面并允许麦克风。
参考：[Microsoft 自签名证书信任说明](https://learn.microsoft.com/en-us/troubleshoot/developer/visualstudio/installation/warnings-untrusted-certificate)。

## 6. Python 接口测试

以下在 **测试客户端的本地仓库 `/home/zj/Qwen3-ASR`** 中执行。
测试脚本只上传音频，不在客户端加载模型，不需要客户端 GPU。

### 6.1 准备依赖和信任证书

```bash
cd /home/zj/Qwen3-ASR
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple httpx numpy soundfile
mkdir -p test_results
scp zenking@192.168.100.230:/data/metahuman_work/qwen3-asr/tls/cert.pem ./cert-230.pem
```

`--ca-cert` 用于校验当前服务器证书，不需要关闭 TLS 校验。

### 6.2 单段音频，模拟实时输入

```bash
python examples/test_streaming_api.py \
  asr_audio_dump_20260513_035431.wav \
  --base-url https://192.168.100.230:8001 \
  --ca-cert ./cert-230.pem \
  --chunk-ms 1000 \
  --output test_results/streaming-single.json
```

测试音频为 6.4 秒、16kHz、单声道。脚本按真实录音速度发送，
输出各分块的累计文本、请求耗时和完成分段数量。

此前该音频识别结果为：

```text
求解不等式x的平方减去3x加2小于0的解。
```

### 6.3 长会话测试

将同一音频重复 40 次，组成 256 秒音频，加速发送：

```bash
python examples/test_streaming_api.py \
  asr_audio_dump_20260513_035431.wav \
  --base-url https://192.168.100.230:8001 \
  --ca-cert ./cert-230.pem \
  --repeat 40 \
  --chunk-ms 2000 \
  --no-realtime \
  --output test_results/streaming-long.json
```

已完成的验证结果：

| 指标 | 结果 |
|---|---|
| 输入音频长度 | 256 秒 |
| 分块请求 | 128 次，全部 HTTP 200 |
| 最终完成分段数 | 9 |
| 上下文长度超限 / HTTP 500 | 未出现 |
| 加速测试墙钟耗时 | 21.782 秒 |
| 首次分块请求耗时 | 2079.1ms，包含首次推理开销 |
| 后续分块请求耗时中位数 | 153.0ms，每次提交 2 秒音频 |

`--no-realtime` 会省略模拟录音等待，256 秒是音频内容长度，而不是实际等待时间。
这项测试验证连续分段与接口稳定性，不是转写准确率或多客户端性能评测。

### 6.4 回归测试

在 **230 宿主机**执行，测试不加载模型：

```bash
docker exec qwen3-asr python \
  /data/shared/Qwen3-ASR/test_streaming_sessions.py \
  /data/shared/Qwen3-ASR/demo_streaming_with_max_len.py
```

6 项测试已通过：大块音频跨段、分块累计与尾音、长度溢出回滚、
结束时溢出恢复、单块无法容纳时返回 422、低能量停顿提前分段。

## 7. HTTP API 协议

服务根地址：`https://192.168.100.230:8001`，无需加 `/v1`。

| 接口 | 方法 | 参数 / 请求体 | 返回 |
|---|---|---|---|
| `/api/start` | POST | 无需请求体 | `session_id` |
| `/api/chunk?session_id=...` | POST | `application/octet-stream`；float32 PCM | 当前累计识别结果 |
| `/api/finish?session_id=...` | POST | 无需请求体 | 最终累计结果，随后释放会话 |

音频块为 **16kHz、单声道、float32 PCM**，归一化至 `[-1, 1]`。
当前部署机器采用小端字节序。上传采样数据，不包含 WAV 文件头。
示例：NumPy 数组转换为 `audio.astype("<f4").tobytes()`。

单个会话内按顺序等待每次分块请求完成，再提交下一块。模型推理使用共享锁串行执行。

结果格式示例（仅展示字段含义）：

```json
{
  "language": "Chinese",
  "text": "已完成分段的文字，加上当前分段的累计识别文字。",
  "segments_completed": 2,
  "segment_seconds": 4.0,
  "total_audio_seconds": 64.0
}
```

`text` 是累计全文，当前未完成分段的末尾文字可能修正。
客户端应替换当前显示文本，直接追加每次响应会造成重复。
完成分段的历史文本被保留，新的分段只把自己的音频与文字前缀送给模型。

会话 10 分钟没有请求后会在后续请求触发的清理中删除。
服务重启会丢失所有内存会话。

## 8. 自动分段的行为与功能边界

原始流式推理每次会重新提交该状态中的全部历史音频，文本前缀也持续增长。
因此长期不结束会话，最终会超过 `8192` token。
`--chunk-size-sec` 控制推理频率，不控制上下文保留长度。

当前修复使用以下方式限制单段状态：

1. 单段达到最大时长时，保存已识别文本并创建新状态。
2. 单段超过最大时长的一半后，最近约 500ms 音频 RMS 小于 `0.005` 时，提前分段。
3. 发生模型输入长度校验错误时，恢复失败前的状态，保留已识别文本，
   在新状态中重新处理尚未识别的音频。
4. 如果单个新状态的音频块仍无法容纳，返回带说明的 HTTP 422。

参数要求：`0 < --chunk-size-sec <= --segment-max-sec <= 30`。
例如将分段上限改为 10 秒，停顿检测会在单段达到 5 秒后启用。
降低分段上限会减少上下文，但可能更频繁地切开连续讲话中的词语。

此处的停顿检测是简单能量判断，没有接入专门的 VAD 模型，也不是语义断句。
接口仍返回累计文字，没有独立的“句子结束”事件。

当前能力边界：

- 支持短音频和长会话的分块输入。
- 底层模型支持 `context` 提示，但当前网页和 `/api/start` 尚未开放热词输入。
- 当前流式接口不返回字词时间戳。
- 当前没有说话人分段标签或多人音轨分离。
- 尚未进行多客户端负载验证；测试结果来自单路输入。

## 9. 音频、结果和日志保存情况

| 数据 | 当前保存情况 |
|---|---|
| 原始 6.4 秒 WAV | 保存在本地仓库根目录 |
| 256 秒测试音频 | 原始 WAV 重复 40 次，在内存拼接，没有另存 WAV |
| 网页麦克风录音 | 没有自动落盘；分段完成或服务重启后，对应音频不会保留 |
| 已识别文字 | 会话内累计在内存；测试脚本用 `--output` 可保存 JSON |
| 服务日志 | 保存于宿主机挂载目录的 `qwen-asr-streaming-https.log` |

先前长会话报告位于本地 `/tmp/qwen-asr-long-streaming-test.json`，
`/tmp` 是临时目录。需要长期保留时，在 **本地仓库**执行：

```bash
mkdir -p test_results
cp /tmp/qwen-asr-long-streaming-test.json test_results/streaming-long-verified.json
```

如果需要保存每次麦克风录音，需要另外增加录音落盘功能，当前服务尚未实现。

## 10. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `navigator.mediaDevices` 或 `getUserMedia` 不可用 | 使用 HTTPS 或 localhost 安全上下文，信任证书并允许麦克风 |
| `NET::ERR_CERT_AUTHORITY_INVALID` | 客户端尚未信任自签名证书，参考第 5 节 |
| 证书日期 / 名称错误 | 检查客户端时间与访问 IP，证书 SAN 应包含访问地址 |
| Python 证书校验失败 | 下载当前 `cert.pem`，通过 `--ca-cert` 指定 |
| `invalid session_id` | 会话已结束、过期或服务重启；重新 `/api/start` |
| decoder prompt 超过 `8192` | 确认运行的是修改后的脚本，并启用了自动分段 |
| HTTP 422，单块无法容纳 | 减小 `--chunk-size-sec`，检查上下文配置 |
| 端口占用 | 检查现有服务实例，停止后再启动 |
| 显存不足 / 显卡不兼容 | 停止继续启动与调参，交由用户处理显存或更换服务器 |
| 重启容器后页面不可访问 | 容器目前不自动启动 ASR；执行第 4 节命令并查看日志 |

当前模型目录已补充 `dtype=bfloat16` 和 tokenizer 的 `fix_mistral_regex=true`，
对应原文件备份在挂载目录 `config-backup-20261012-115840/`。
如果替换或重新下载模型目录，需要重新核对这些配置。
