"""
用 MiniMax 合成一段语音，或列出可用音色。

  python3 tts_say.py "小朋友，抬头看看天空。" -o hello.mp3
  python3 tts_say.py "千江有水千江月。" -o zen.mp3 --voice "Chinese (Mandarin)_Radio_Host" --speed 0.82
  python3 tts_say.py --list-voices            # 列出普通话系统音色

密钥读取顺序：环境变量 MINIMAX_API_KEY → MINIMAX_KEY_FILE 指定的文件 → 仓库根目录的 .minimax_key。
"""
import argparse
import json
import sys
import urllib.request

import vidkit as vk


def list_voices(key, host):
    req = urllib.request.Request(f"https://{host}/v1/get_voice", data=json.dumps({"voice_type": "system"}).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    res = json.loads(urllib.request.urlopen(req, timeout=60).read())
    if res.get("base_resp", {}).get("status_code", 0) != 0:
        sys.exit(f"查询失败：{res['base_resp']}")
    for v in res.get("system_voice", []):
        vid = v["voice_id"]
        if vid.startswith(("Chinese (Mandarin)", "male-", "female-", "presenter", "audiobook")):
            print(f"{vid:45s} {v.get('voice_name', '')}  {''.join(v.get('description') or [])[:40]}")


def main():
    ap = argparse.ArgumentParser(description="MiniMax 语音合成")
    ap.add_argument("text", nargs="?", help="要朗读的文字；可用 <#0.5#> 插入 0.5 秒停顿")
    ap.add_argument("-o", "--out", default="tts.mp3")
    ap.add_argument("--voice", default="Chinese (Mandarin)_Gentle_Senior")
    ap.add_argument("--model", default="speech-2.6-hd")
    ap.add_argument("--speed", type=float, default=1.0, help="语速，0.5–2.0")
    ap.add_argument("--emotion", default=None, help="happy / sad / angry / fearful / disgusted / surprised / neutral")
    ap.add_argument("--host", default="api.minimaxi.com")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args()

    key = vk.minimax_key()
    if not key:
        sys.exit("未找到密钥：请设置 MINIMAX_API_KEY，或把密钥写入仓库根目录的 .minimax_key 文件")
    if a.list_voices:
        return list_voices(key, a.host)
    if not a.text:
        ap.error("请提供要朗读的文字")
    vk.MiniMaxTTS(key, a.voice, a.model, a.speed, a.emotion, hosts=(a.host,))(a.text, a.out)
    print("已生成：", a.out)


if __name__ == "__main__":
    main()
