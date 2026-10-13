#!/usr/bin/env python3
"""分块调用 Qwen3-ASR 流式 HTTP API，按音频时长模拟实时输入。

    python examples/test_streaming_api.py
    python examples/test_streaming_api.py audio.wav --chunk-ms 500

依赖：pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple httpx numpy soundfile
"""

import argparse
import json
from pathlib import Path
import ssl
import sys
import time

import httpx
import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1] / "asr_audio_dump_20260513_035431.wav",
    )
    parser.add_argument("--base-url", default="https://192.168.100.230:8001")
    parser.add_argument("--ca-cert", type=Path, help="信任的自签名证书或 CA 证书路径")
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=1, help="重复音频次数，测试长会话")
    parser.add_argument("--no-realtime", action="store_true", help="不等待录音时长，加速长音频测试")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, help="保存完整测试记录为 JSON")
    args = parser.parse_args()
    if args.chunk_ms <= 0 or args.timeout <= 0 or args.repeat <= 0:
        parser.error("--chunk-ms、--timeout 和 --repeat 必须大于 0")
    if not args.audio.is_file():
        parser.error(f"找不到音频文件：{args.audio}")
    if args.ca_cert and not args.ca_cert.is_file():
        parser.error(f"找不到证书文件：{args.ca_cert}")

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        parser.error(f"音频需要 16000 Hz，当前为 {sample_rate} Hz；请先重采样")
    audio = np.clip(audio.mean(axis=1), -1.0, 1.0).astype("<f4")
    if audio.size == 0:
        parser.error("音频为空")
    audio = np.tile(audio, args.repeat)
    chunk_samples = max(1, round(sample_rate * args.chunk_ms / 1000))
    base_url = args.base_url.rstrip("/")
    events = []
    report = {"audio": str(args.audio.resolve()), "base_url": base_url,
              "duration_sec": audio.size / sample_rate, "chunk_ms": args.chunk_ms,
              "chunks": events}
    print(f"服务：{base_url}，音频：{report['duration_sec']:.2f} 秒，分块：{args.chunk_ms} ms", flush=True)

    def post(client, path, **kwargs):
        response = client.post(base_url + path, **kwargs)
        response.raise_for_status()
        return response.json()

    try:
        verify = ssl.create_default_context(cafile=str(args.ca_cert)) if args.ca_cert else True
        with httpx.Client(timeout=args.timeout, verify=verify) as client:
            session_id = post(client, "/api/start")["session_id"]
            report["session_id"] = session_id
            started = time.perf_counter()
            try:
                for offset in range(0, audio.size, chunk_samples):
                    chunk = audio[offset:offset + chunk_samples]
                    audio_sec = (offset + chunk.size) / sample_rate
                    # 等待该段音频被“录制”完成，推理耗时计入实时进度。
                    if not args.no_realtime:
                        time.sleep(max(0.0, started + audio_sec - time.perf_counter()))
                    sent = time.perf_counter()
                    result = post(
                        client, "/api/chunk", params={"session_id": session_id},
                        content=chunk.tobytes(),
                        headers={"Content-Type": "application/octet-stream"},
                    )
                    latency_ms = (time.perf_counter() - sent) * 1000
                    event = {"audio_sec": audio_sec, "request_ms": round(latency_ms, 1), **result}
                    events.append(event)
                    segment = result.get('segments_completed', 0)
                    print(f"[{audio_sec:4.1f}s / {latency_ms:6.1f}ms / 完成段数 {segment}] {result['text'][-100:]}", flush=True)
            finally:
                # 即使某个分块失败，也尝试释放已创建的服务端会话。
                final = post(client, "/api/finish", params={"session_id": session_id})
            report["final"] = final
            report["elapsed_sec"] = round(time.perf_counter() - started, 3)
    except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
        print(f"测试失败：{exc}", file=sys.stderr)
        return 1

    print(f"最终结果（{final['language']}）：{final['text']}")
    print(f"总耗时：{report['elapsed_sec']:.3f} 秒（{'加速输入' if args.no_realtime else '包含实时音频输入等待'}）")
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"测试记录：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
