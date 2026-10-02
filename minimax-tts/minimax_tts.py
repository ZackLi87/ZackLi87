#!/usr/bin/env python3
"""
MiniMax 语音合成（T2A v2）独立调用脚本，仅依赖 Python 标准库（Python 3.8+）。

密钥读取顺序：
  1. 命令行参数 --key-file 指定的文件
  2. 环境变量 MINIMAX_API_KEY
  3. 与本脚本同目录的 minimax_key.txt（请勿提交到 git）

示例：
  python3 minimax_tts.py "你好，世界。" -o hello.mp3
  python3 minimax_tts.py -f 文稿.txt -o 文稿.mp3 --voice "Chinese (Mandarin)_Radio_Host" --speed 0.9
  python3 minimax_tts.py --list-voices
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KEY_FILE = os.path.join(HERE, "minimax_key.txt")


def load_key(key_file=None):
    if key_file:
        with open(key_file, encoding="utf-8") as f:
            return f.read().strip()
    if os.environ.get("MINIMAX_API_KEY", "").strip():
        return os.environ["MINIMAX_API_KEY"].strip()
    if os.path.exists(DEFAULT_KEY_FILE):
        with open(DEFAULT_KEY_FILE, encoding="utf-8") as f:
            return f.read().strip()
    sys.exit("未找到密钥：请把密钥粘贴到 minimax_key.txt，或设置环境变量 MINIMAX_API_KEY")


def post(host, path, key, payload):
    req = urllib.request.Request(f"https://{host}{path}", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        sys.exit(f"无法连接 {host}：{e}")


def synthesize(text, out, key, voice="Chinese (Mandarin)_Gentle_Senior", model="speech-2.6-hd",
               speed=1.0, vol=1.0, pitch=0, emotion=None, fmt="mp3", host="api.minimaxi.com"):
    """合成语音并写入文件。text 中可用 <#秒数#> 插入停顿，例如 “你好<#0.5#>世界”。"""
    voice_setting = {"voice_id": voice, "speed": speed, "vol": vol, "pitch": pitch}
    if emotion:
        voice_setting["emotion"] = emotion
    res = post(host, "/v1/t2a_v2", key, {
        "model": model, "text": text, "stream": False, "language_boost": "auto",
        "voice_setting": voice_setting,
        "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": fmt, "channel": 1},
    })
    base = res.get("base_resp", {})
    if base.get("status_code", 0) != 0 or not res.get("data", {}).get("audio"):
        sys.exit(f"合成失败：状态码 {base.get('status_code')}，{base.get('status_msg')}")
    with open(out, "wb") as f:
        f.write(bytes.fromhex(res["data"]["audio"]))
    info = res.get("extra_info", {})
    return info.get("audio_length", 0) / 1000, info.get("usage_characters")


def list_voices(key, host="api.minimaxi.com", keyword=None):
    res = post(host, "/v1/get_voice", key, {"voice_type": "system"})
    base = res.get("base_resp", {})
    if base.get("status_code", 0) != 0:
        sys.exit(f"查询失败：状态码 {base.get('status_code')}，{base.get('status_msg')}")
    for v in res.get("system_voice", []):
        line = f"{v['voice_id']:48s} {v.get('voice_name', '')}  {''.join(v.get('description') or [])[:50]}"
        if keyword is None or keyword in line:
            print(line)


def main():
    ap = argparse.ArgumentParser(description="MiniMax 语音合成")
    ap.add_argument("text", nargs="?", help="要朗读的文字")
    ap.add_argument("-f", "--file", help="从文本文件读取要朗读的文字（UTF-8）")
    ap.add_argument("-o", "--out", default="output.mp3", help="输出文件，默认 output.mp3")
    ap.add_argument("--voice", default="Chinese (Mandarin)_Gentle_Senior", help="音色 ID")
    ap.add_argument("--model", default="speech-2.6-hd", help="模型，如 speech-2.6-hd、speech-02-hd")
    ap.add_argument("--speed", type=float, default=1.0, help="语速 0.5–2.0")
    ap.add_argument("--vol", type=float, default=1.0, help="音量 0–10")
    ap.add_argument("--pitch", type=int, default=0, help="音调 -12–12")
    ap.add_argument("--emotion", help="情绪：happy / sad / angry / fearful / disgusted / surprised / neutral")
    ap.add_argument("--format", default="mp3", choices=["mp3", "wav", "flac", "pcm"])
    ap.add_argument("--host", default="api.minimaxi.com", help="国内站 api.minimaxi.com；国际站 api.minimax.io")
    ap.add_argument("--key-file", help="密钥文件路径")
    ap.add_argument("--list-voices", nargs="?", const="", metavar="关键词", help="列出系统音色，可按关键词筛选")
    a = ap.parse_args()

    key = load_key(a.key_file)
    if a.list_voices is not None:
        return list_voices(key, a.host, a.list_voices or None)
    text = open(a.file, encoding="utf-8").read().strip() if a.file else a.text
    if not text:
        ap.error("请提供要朗读的文字，或用 -f 指定文本文件")
    sec, chars = synthesize(text, a.out, key, a.voice, a.model, a.speed, a.vol, a.pitch, a.emotion, a.format, a.host)
    print(f"已生成 {a.out}（{sec:.1f} 秒，计费字符 {chars}）")


if __name__ == "__main__":
    main()
