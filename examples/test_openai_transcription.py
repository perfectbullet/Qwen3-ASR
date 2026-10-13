#!/usr/bin/env python3
"""通过 OpenAI 转写 API 测试已部署的 Qwen3-ASR 服务。

在仓库根目录运行：
    python examples/test_openai_transcription.py
    python examples/test_openai_transcription.py /path/to/audio.wav

依赖安装：
    pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple openai
"""

import argparse
import os
from pathlib import Path
import sys
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "asr_audio_dump_20260513_035431.wav",
        help="本地音频路径，默认使用仓库中的测试 WAV",
    )
    parser.add_argument("--base-url", default="http://192.168.100.230:8001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--timeout", type=float, default=120.0, help="请求超时秒数")
    args = parser.parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        parser.error(f"找不到音频文件：{audio_path}")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")

    print(f"服务地址：{args.base_url}")
    print(f"模型名称：{args.model}")
    print(f"音频文件：{audio_path}")
    try:
        with OpenAI(
            base_url=args.base_url,
            api_key=os.environ.get("ASR_API_KEY", "EMPTY"),
            timeout=args.timeout,
            max_retries=0,
        ) as client:
            with audio_path.open("rb") as audio_file:
                started = time.perf_counter()
                transcription = client.audio.transcriptions.create(
                    model=args.model,
                    file=audio_file,
                    response_format="json",
                    temperature=0.0,
                )
                elapsed = time.perf_counter() - started
    except APITimeoutError:
        print("转写请求超时，可增大 --timeout。", file=sys.stderr)
        return 1
    except APIConnectionError as exc:
        print(f"连接失败，请检查服务和 --base-url：{exc}", file=sys.stderr)
        return 1
    except APIStatusError as exc:
        print(f"转写失败（HTTP {exc.status_code}）：{exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"读取音频失败：{exc}", file=sys.stderr)
        return 1

    print(f"请求耗时：{elapsed:.3f} 秒")
    print(f"原始转写：{transcription.text}")
    # 当前服务可能在 text 中保留 Qwen 的语言前缀和分隔符。
    prefix, separator, text = transcription.text.partition("<asr_text>")
    if separator:
        if prefix.startswith("language "):
            print(f"识别语言：{prefix.removeprefix('language ').strip()}")
        print(f"转写文本：{text.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
