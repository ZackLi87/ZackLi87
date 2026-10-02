"""
vidkit：竖屏科普短视频的通用生成流程。

各期视频只需提供：分镜文案（SCENES）、逐帧绘制函数、背景音乐函数；
语音合成、时间轴排布、混音、多进程渲染与编码由本模块完成。

语音：读取到 MiniMax 密钥（环境变量 MINIMAX_API_KEY 或仓库根目录的 .minimax_key 文件）时使用
MiniMax T2A（api.minimaxi.com），否则回退到离线 sherpa-onnx 模型。调用方法见 minimax-tts/README.md。
"""
import hashlib
import json
import os
import re
import subprocess
import urllib.request
import wave
from multiprocessing import Pool

import numpy as np

SR = 44100


# ---------------------------------------------------------------- 语音合成
def to_samples(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def trim(x):
    nz = np.where(np.abs(x) > 0.01)[0]
    return x[max(nz[0] - 600, 0): nz[-1] + 2500] if len(nz) else x


class MiniMaxTTS:
    ext = ".mp3"

    def __init__(self, key, voice, model="speech-2.6-hd", speed=1.0, emotion=None,
                 hosts=("api.minimaxi.com", "api.minimax.io")):
        self.key, self.voice, self.model, self.speed, self.emotion = key, voice, model, speed, emotion
        self.hosts = [os.environ["MINIMAX_API_HOST"]] if os.environ.get("MINIMAX_API_HOST") else list(hosts)
        self.tag = f"minimax|{voice}|{model}|{speed}|{emotion}"

    def __call__(self, text, out):
        vs = {"voice_id": self.voice, "speed": self.speed, "vol": 1.0, "pitch": 0}
        if self.emotion:
            vs["emotion"] = self.emotion
        body = json.dumps({
            "model": self.model, "text": text, "stream": False, "language_boost": "Chinese",
            "voice_setting": vs,
            "audio_setting": {"sample_rate": 44100, "bitrate": 128000, "format": "mp3", "channel": 1},
        }).encode()
        err = None
        for host in self.hosts:
            req = urllib.request.Request(f"https://{host}/v1/t2a_v2", data=body, headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    res = json.loads(r.read())
            except Exception as e:
                err = e
                continue
            base = res.get("base_resp", {})
            if base.get("status_code", 0) != 0 or not res.get("data", {}).get("audio"):
                err = RuntimeError(f"{host}: {base}")
                continue
            with open(out, "wb") as f:
                f.write(bytes.fromhex(res["data"]["audio"]))
            return
        raise RuntimeError(f"MiniMax 语音合成失败：{err}")


class LocalTTS:
    ext = ".wav"

    def __init__(self, assets, speed=1.0):
        import sherpa_onnx
        d = os.path.join(assets, "matcha-icefall-zh-baker")
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=f"{d}/model-steps-3.onnx",
                    vocoder=os.path.join(assets, "vocos-22khz-univ.onnx"),
                    lexicon=f"{d}/lexicon.txt", tokens=f"{d}/tokens.txt", dict_dir=f"{d}/dict"),
                num_threads=4),
            rule_fsts=f"{d}/phone.fst,{d}/date.fst,{d}/number.fst")
        self.eng, self.speed, self.tag = sherpa_onnx.OfflineTts(cfg), speed, f"local|{speed}"

    def __call__(self, text, out):
        a = self.eng.generate(re.sub(r"<#[\d.]+#>", "", text), sid=0, speed=self.speed)
        s = (np.clip(np.array(a.samples), -1, 1) * 32767).astype(np.int16)
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(a.sample_rate)
            w.writeframes(s.tobytes())


KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".minimax_key")
KEY_FILE2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimax-tts", "minimax_key.txt")


def minimax_key():
    """读取 MiniMax 密钥：优先环境变量 MINIMAX_API_KEY，其次 MINIMAX_KEY_FILE 指定的文件，
    再次是仓库根目录的 .minimax_key，最后是 minimax-tts/minimax_key.txt（均已列入 .gitignore）。"""
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if key:
        return key
    for path in (os.environ.get("MINIMAX_KEY_FILE"), KEY_FILE, KEY_FILE2):
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
    return ""


def make_tts(assets, voice, model="speech-2.6-hd", speed=1.0, emotion=None):
    key = minimax_key()
    if key:
        return MiniMaxTTS(key, voice, model, speed, emotion)
    print("未设置 MINIMAX_API_KEY，使用离线语音模型")
    return LocalTTS(assets, speed)


def synth_lines(scenes, tts, cache):
    """逐句合成旁白并缓存，返回 [(scene_idx, line_dict, samples)]。"""
    os.makedirs(cache, exist_ok=True)
    out = []
    for si, sc in enumerate(scenes):
        for ln in sc["lines"]:
            say = ln.get("say", ln["show"])
            h = hashlib.md5(f"{tts.tag}|{say}".encode()).hexdigest()
            npy = os.path.join(cache, h + ".npy")
            if not os.path.exists(npy):
                tmp = os.path.join(cache, h + tts.ext)
                tts(say, tmp)
                np.save(npy, trim(to_samples(tmp)))
                os.remove(tmp)
            out.append((si, ln, np.load(npy)))
    return out


def build_timeline(scenes, voice, lead=0.4, gap=0.3, tail=0.6, final_hold=2.5):
    """按语音时长排布：每句 t0/t1，每个场景 start/end（秒）。"""
    tl, t = [], 0.0
    for si, sc in enumerate(scenes):
        start, cur, lines = t, t + lead, []
        for vsi, ln, x in voice:
            if vsi != si:
                continue
            d = len(x) / SR
            lines.append(dict(ln, t0=cur, t1=cur + d, audio=x))
            cur += d + gap + ln.get("pause", 0.0)
        end = cur - gap + sc.get("tail", final_hold if si == len(scenes) - 1 else tail)
        tl.append(dict(sc, start=start, end=end, lines=lines))
        t = end
    return tl, t


def strip_audio(tl):
    """去掉音频数组，便于传给渲染子进程。"""
    return [dict(s, lines=[{k: v for k, v in l.items() if k != "audio"} for l in s["lines"]]) for s in tl]


# ---------------------------------------------------------------- 音频
def pluck(freq, dur, decay=0.996, seed=0):
    """Karplus–Strong 拨弦。"""
    n, N = int(dur * SR), max(int(SR / freq), 2)
    rng = np.random.default_rng(int(freq * 10) + seed)
    y = np.zeros(n + N + 2)
    y[1:N + 1] = rng.uniform(-1, 1, N) * np.hanning(N)
    for k in range(1, (n // N) + 1):
        a, b = 1 + k * N, min(1 + (k + 1) * N, n + N + 2)
        y[a:b] = decay * 0.5 * (y[a - N:b - N] + y[a - N - 1:b - N - 1])
    return y[1:n + 1] * np.minimum(1, np.arange(n) / 80)


def write_mix(tl, total, music, path, sfx=None):
    """旁白 + 背景音乐（music(n_samples) → array）+ 可选音效，写出 16 bit 单声道 wav。"""
    n = int(total * SR) + SR
    voice = np.zeros(n)
    for sc in tl:
        for ln in sc["lines"]:
            s = int(ln["t0"] * SR)
            voice[s:s + len(ln["audio"])] += ln["audio"]
    voice *= 0.89 / (np.max(np.abs(voice)) + 1e-9)
    mix = voice + music(n)
    if sfx is not None:
        mix = mix + sfx(n)
    mix = np.clip(mix, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- 渲染与编码
def encode(render, nframes, wav, out, size, fps=30, init=None, initargs=()):
    """多进程逐帧渲染（render(i) → RGB bytes），经 ffmpeg 编码并做响度标准化。"""
    w, h = size
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-i", wav,
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-af", "loudnorm=I=-15:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
         "-movflags", "+faststart", "-shortest", out], stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init, initargs=initargs) as pool:
        for k, buf in enumerate(pool.imap(render, range(nframes), chunksize=8)):
            ff.stdin.write(buf)
            if k % 300 == 0:
                print(f"  帧 {k}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()


# ---------------------------------------------------------------- 动画工具
def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def fade(t, t0, dur=0.5):
    return ease((t - t0) / dur)


def pop(t, t0, dur=0.45):
    """带轻微回弹的出现动画，返回缩放系数 0 → 1。"""
    x = max(0.0, min(1.0, (t - t0) / dur))
    return 1 + 2.7 * (x - 1) ** 3 + 1.7 * (x - 1) ** 2 if x < 1 else 1.0


def lerp_col(c1, c2, u):
    u = max(0.0, min(1.0, u))
    return tuple(int(round(a + (b - a) * u)) for a, b in zip(c1, c2))


def wrap_balanced(s, measure, maxw, punct="，、：；"):
    """一行放不下时，在靠近中点的标点处断为两行。measure(str) → 像素宽度。"""
    if measure(s) <= maxw:
        return [s]
    mid = len(s) / 2
    cands = [i + 1 for i, ch in enumerate(s[:-1]) if ch in punct]
    best = min(cands, key=lambda i: abs(i - mid), default=None)
    k = best if best is not None and abs(best - mid) <= len(s) * 0.25 else round(mid)
    return [s[:k], s[k:]]
