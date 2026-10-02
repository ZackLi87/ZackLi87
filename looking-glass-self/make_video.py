"""
生成竖屏短视频《照见》：以禅意的方式重述
“别人眼中的自己不是真的自己，我眼中的别人眼中的自己才是真的自己”。

流程：语音合成（优先 MiniMax T2A，未配置密钥时回退到离线 sherpa-onnx 模型）
→ 按语音时长排布时间轴 → Pillow 逐帧绘制水墨风格画面（2 倍超采样）
→ 程序合成古琴式拨弦与颂钵背景音 → ffmpeg 封装 H.264/AAC 并做响度标准化。

用法：
  export MINIMAX_API_KEY=...            # 可选；不设置则使用离线语音
  python3 make_video.py [--assets DIR] [--out OUT.mp4] [--cover COVER.png]
  python3 make_video.py --preview 12.5  # 仅导出第 12.5 秒的单帧
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import urllib.request
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, S = 1080, 1920, 30, 2      # 输出分辨率、帧率、超采样倍数
SR = 44100

PAPER = (241, 236, 226)               # 宣纸底色
INK = (36, 33, 30)                    # 浓墨
GREY = (128, 120, 110)                # 淡墨
FAINT = (205, 197, 184)               # 极淡墨
SEAL = (176, 48, 38)                  # 朱砂印

LEAD, GAP, TAIL, FINAL_HOLD = 0.55, 0.55, 0.8, 3.2
TEXT_Y = 1330                         # 文案区中心

# ---------------------------------------------------------------- 文案
# show：画面文字；say：旁白（缺省同 show，可含 MiniMax 停顿标记 <#秒#>）；
# keep：下一句出现时保留并与之上下排列；
# note：出处等小字。
SCENES = [
    dict(key="enso", lines=[
        dict(show="有人说，"),
        dict(show="别人眼中的你，不是真的你；", keep=True),
        dict(show="你眼中的别人眼中的你，才是。"),
    ]),
    dict(key="moon", lines=[
        dict(show="千江有水千江月。", note="—— 禅家语"),
        dict(show="一千双眼睛里，有一千个你。"),
        dict(show="那些你，你一个也看不见。"),
        dict(show="你看见的，只是自己以为的那一个。"),
    ]),
    dict(key="daily", lines=[
        dict(show="出门前换下的衣裳，", keep=True),
        dict(show="发出前删去的那句话，"),
        dict(show="都在回应那个“以为”。"),
    ]),
    dict(key="flag", lines=[
        dict(show="不是风动，不是幡动，", say="不是风动，<#0.3#>不是幡动，", keep=True),
        dict(show="仁者心动。", say="仁者<#0.25#>心动。", note="——《六祖坛经》"),
        dict(show="照见你的，从来是你自己的心。"),
    ]),
    dict(key="mirror", lines=[
        dict(show="心镜蒙尘，处处都是审视；", keep=True),
        dict(show="拂去尘埃，"),
        dict(show="别人如何看你，便只是别人的事。"),
        dict(show="愿你照见，本来面目。", say="愿你照见，<#0.4#>本来面目。", final=True),
    ]),
]

ARGS = None
GA = 1.0


# ---------------------------------------------------------------- 语音合成
def to_samples(path):
    """用 ffmpeg 将任意音频解码为 SR 采样率的单声道 float32。"""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def trim(x):
    nz = np.where(np.abs(x) > 0.01)[0]
    return x[max(nz[0] - 600, 0): nz[-1] + 2500] if len(nz) else x


class MiniMaxTTS:
    def __init__(self, key, voice, model, speed, hosts):
        self.key, self.voice, self.model, self.speed, self.hosts = key, voice, model, speed, hosts
        self.tag = f"minimax|{voice}|{model}|{speed}"

    def __call__(self, text, out_mp3):
        body = json.dumps({
            "model": self.model, "text": text, "stream": False, "language_boost": "Chinese",
            "voice_setting": {"voice_id": self.voice, "speed": self.speed, "vol": 1.0, "pitch": 0},
            "audio_setting": {"sample_rate": 44100, "bitrate": 128000, "format": "mp3", "channel": 1},
        }).encode()
        err = None
        for host in self.hosts:
            req = urllib.request.Request(f"https://{host}/v1/t2a_v2", data=body, headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    res = json.loads(r.read())
            except Exception as e:      # 网络不可达时尝试下一个地址
                err = e
                continue
            base = res.get("base_resp", {})
            if base.get("status_code", 0) != 0 or not res.get("data", {}).get("audio"):
                err = RuntimeError(f"{host}: {base}")
                continue
            with open(out_mp3, "wb") as f:
                f.write(bytes.fromhex(res["data"]["audio"]))
            return
        raise RuntimeError(f"MiniMax 语音合成失败：{err}")


class LocalTTS:
    def __init__(self, assets, speed):
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

    def __call__(self, text, out_wav):
        a = self.eng.generate(text, sid=0, speed=self.speed)
        s = (np.clip(np.array(a.samples), -1, 1) * 32767).astype(np.int16)
        with wave.open(out_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(a.sample_rate)
            w.writeframes(s.tobytes())


def make_tts(args):
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if key:
        hosts = [h for h in [os.environ.get("MINIMAX_API_HOST")] if h] or \
            ["api.minimaxi.com", "api.minimax.io"]
        return MiniMaxTTS(key, args.voice, args.model, args.speed, hosts)
    print("未设置 MINIMAX_API_KEY，使用离线语音模型")
    return LocalTTS(args.assets, 0.95)


def synth_lines(args):
    cache = os.path.join(args.assets, "tts_cache")
    os.makedirs(cache, exist_ok=True)
    tts, out = make_tts(args), []
    for si, sc in enumerate(SCENES):
        for ln in sc["lines"]:
            say = ln.get("say", ln["show"])
            if not isinstance(tts, MiniMaxTTS):
                say = re.sub(r"<#[\d.]+#>", "", say)
            h = hashlib.md5(f"{tts.tag}|{say}".encode()).hexdigest()
            npy = os.path.join(cache, h + ".npy")
            if not os.path.exists(npy):
                tmp = os.path.join(cache, h + (".mp3" if isinstance(tts, MiniMaxTTS) else ".wav"))
                tts(say, tmp)
                np.save(npy, trim(to_samples(tmp)))
                os.remove(tmp)
            out.append((si, ln, np.load(npy)))
    return out


def build_timeline(voice):
    scenes, t = [], 0.0
    for si, sc in enumerate(SCENES):
        start, cur, lines = t, t + LEAD, []
        for vsi, ln, x in voice:
            if vsi != si:
                continue
            d = len(x) / SR
            lines.append(dict(ln, t0=cur, t1=cur + d, audio=x))
            cur += d + GAP + (0.35 if ln.get("note") else 0)
        end = cur - GAP + (FINAL_HOLD if si == len(SCENES) - 1 else TAIL)
        scenes.append(dict(sc, start=start, end=end, lines=lines))
        t = end
    return scenes, t


# ---------------------------------------------------------------- 背景音
def pluck(freq, dur, decay=0.996):
    """Karplus–Strong 拨弦，音色近似古琴。按周期分块向量化计算。"""
    n, N = int(dur * SR), max(int(SR / freq), 2)
    rng = np.random.default_rng(int(freq * 10))
    y = np.zeros(n + N + 2)                    # y[0] 为补零，激励位于 y[1:N+1]
    y[1:N + 1] = rng.uniform(-1, 1, N) * np.hanning(N)
    for k in range(1, (n // N) + 1):
        a, b = 1 + k * N, min(1 + (k + 1) * N, n + N + 2)
        y[a:b] = decay * 0.5 * (y[a - N:b - N] + y[a - N - 1:b - N - 1])
    out = y[1:n + 1]
    return out * np.minimum(1, np.arange(n) / 80)


def bowl(dur, f0=196.0):
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for ratio, amp, tau in [(1, 1.0, 6.0), (2.76, 0.5, 3.5), (5.40, 0.25, 1.8), (8.93, 0.12, 0.9)]:
        f = f0 * ratio
        out += amp * np.exp(-t / tau) * (np.sin(2 * np.pi * f * t) + 0.6 * np.sin(2 * np.pi * (f + 1.3) * t))
    return out * np.minimum(1, t / 0.01)


def make_music(total, final_t):
    n = int(total * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    # 低沉的持续音
    out += 0.05 * (np.sin(2 * np.pi * 73.4 * t) + 0.6 * np.sin(2 * np.pi * 110 * t)) * \
        (0.7 + 0.3 * np.sin(2 * np.pi * t / 9))
    # 稀疏的五声音阶拨弦（D 羽调式）
    scale = [146.8, 174.6, 196.0, 220.0, 261.6, 293.7, 349.2, 392.0]
    rng = np.random.default_rng(7)
    tt = 1.2
    while tt < total - 3:
        for j, f in enumerate([scale[rng.integers(0, 6)]] + ([scale[rng.integers(3, 8)]] if rng.random() < 0.35 else [])):
            p = pluck(f, 3.5)
            s = int((tt + j * 0.18) * SR)
            e = min(s + len(p), n)
            out[s:e] += 0.32 * p[:e - s]
        tt += rng.choice([2.4, 3.2, 4.0])
    for at in (0.0, final_t):              # 开篇与结尾的颂钵
        b = bowl(min(8.0, total - at))
        s = int(at * SR)
        out[s:s + len(b)] += 0.22 * b[:n - s]
    out /= np.max(np.abs(out)) + 1e-9
    return out * np.minimum(1, np.minimum(t / 1.0, (total - t) / 2.5)) * 0.16


def make_audio(scenes, total, path):
    n = int(total * SR) + SR
    voice = np.zeros(n)
    for sc in scenes:
        for ln in sc["lines"]:
            s = int(ln["t0"] * SR)
            voice[s:s + len(ln["audio"])] += ln["audio"]
    voice *= 0.89 / (np.max(np.abs(voice)) + 1e-9)
    final_t = scenes[-1]["lines"][-1]["t0"] - 0.3
    mix = np.clip(voice + make_music(n / SR, final_t), -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- 绘图工具
_fonts = {}


def F(kind, size):
    key = (kind, size)
    if key not in _fonts:
        name = {"serif": "NotoSerifSC-Medium.otf", "light": "NotoSerifSC-Regular.otf",
                "bold": "NotoSerifSC-Bold.otf"}[kind]
        _fonts[key] = ImageFont.truetype(os.path.join(ARGS.assets, "fonts", name), size * S)
    return _fonts[key]


def C(col, a=1.0):
    a = max(0.0, min(1.0, a * GA))
    return tuple(int(PAPER[i] + (col[i] - PAPER[i]) * a) for i in range(3))


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fade(t, t0, dur=0.8):
    return ease((t - t0) / dur)


def P(*v):
    return [int(round(u * S)) for u in v]


def text(d, s, x, y, size, col=INK, a=1.0, kind="serif", anchor="mm", bleed=True):
    """墨迹晕开式出现：未完全显现时，外圈有一层随之收拢的淡墨。"""
    if a <= 0.01:
        return
    f = F(kind, size)
    if bleed and a < 0.999:
        d.text(P(x, y), s, font=f, fill=C(col, 0.22 * a), anchor=anchor,
               stroke_width=max(1, round(5 * S * (1 - a))), stroke_fill=C(col, 0.12 * a))
    d.text(P(x, y), s, font=f, fill=C(col, a), anchor=anchor)


def vtext(d, s, x, y, size, col=INK, a=1.0, kind="serif", gap=1.08):
    """竖排文字，(x, y) 为首字中心。"""
    for i, ch in enumerate(s):
        text(d, ch, x, y + i * size * gap, size, col, a, kind)


_enso_cache = {}


def enso(d, cx, cy, r, prog, col=INK, a=1.0, w=30, seed=0, start=-105):
    """书法“圆相”：起笔饱满，收笔渐细并带飞白。prog ∈ [0, 1] 为书写进度。"""
    if a <= 0.01 or prog <= 0:
        return
    key = (cx, cy, r, w, seed, start)
    if key not in _enso_cache:
        rng = np.random.default_rng(seed)
        n = 520
        pts = []
        for i in range(n):
            u = i / (n - 1)
            th = math.radians(start + 335 * u)
            rr = r * (1 + 0.018 * math.sin(3 * th + seed) + 0.01 * math.sin(7 * th))
            width = w * (0.8 + 0.25 * math.sin(math.pi * min(u * 1.3, 1))) * (1 - 0.78 * u ** 2.2)
            x, y = cx + rr * math.cos(th), cy + rr * math.sin(th)
            nx, ny = math.cos(th), math.sin(th)
            # 飞白：收笔段由若干“笔毛”组成，随机断开
            dry = max(0.0, (u - 0.55) / 0.45)
            bristles = []
            for b in np.linspace(-0.5, 0.5, 7):
                if rng.random() > dry * 0.75:
                    bristles.append((x + nx * b * width, y + ny * b * width,
                                     width / 7 * (1.1 + 0.4 * rng.random())))
            pts.append((u, x, y, width / 2, dry, bristles))
        _enso_cache[key] = pts
    for u, x, y, hw, dry, bristles in _enso_cache[key]:
        if u > prog:
            break
        if dry < 0.15:
            d.ellipse(P(x - hw, y - hw, x + hw, y + hw), fill=C(col, a))
        else:
            for bx, by, br in bristles:
                d.ellipse(P(bx - br, by - br, bx + br, by + br), fill=C(col, a * (1 - 0.25 * dry)))


def stroke(d, pts, col, a, width):
    if a > 0.01 and len(pts) > 1:
        d.line([tuple(P(*p)) for p in pts], fill=C(col, a), width=max(1, round(width * S)), joint="curve")


def seal(d, s, x, y, size, a=1.0):
    """朱文方印（两字竖排）。"""
    if a <= 0.01:
        return
    sc = 1 + 0.25 * (1 - a)
    h = size * 1.15 * sc
    d.rounded_rectangle(P(x - h / 2, y - h, x + h / 2, y + h), radius=6 * S, fill=C(SEAL, a))
    for i, ch in enumerate(s):
        d.text(P(x, y - h / 2 + i * h), ch, font=F("bold", int(size * sc)), fill=C(PAPER, 1.0)
               if a > 0.98 else C(SEAL, a * 0.2), anchor="mm")


# ---------------------------------------------------------------- 各场景画面
def sc_enso(d, t, ts):
    enso(d, 540, 720, 250, 0.3 + t / 2.0, INK, 1.0, 34, seed=1)
    enso(d, 600, 690, 165, (t - ts[1]) / 2.0, GREY, 0.7, 18, seed=2, start=60)
    enso(d, 500, 760, 95, (t - ts[2]) / 1.8, SEAL, 0.75, 12, seed=3, start=200)


def sc_moon(d, t, ts):
    a = fade(t, 0, 1.2)
    d.ellipse(P(540 - 92, 410 - 92, 540 + 92, 410 + 92), fill=C(FAINT, 0.55 * a), outline=C(INK, a),
              width=3 * S)
    rows = [650, 755, 860, 965, 1070]
    mist = fade(t, ts[2], 1.2)
    clear = fade(t, ts[3], 1.0)
    for k, y in enumerate(rows):
        ak = fade(t, ts[0] + 0.25 + 0.35 * k if k < 2 else ts[1] + 0.3 * (k - 2), 1.0)
        mine = k == 4
        # 水波
        pts = [(x, y + 26 + 4 * math.sin(x / 38 + t * 1.3 + k)) for x in range(110, 971, 12)]
        stroke(d, pts, GREY, ak * 0.55, 2)
        # 倒影：被水波打碎的月
        ra = ak * (1 - 0.75 * mist) if not mine else ak * (1 - 0.75 * mist + 0.75 * clear)
        col = INK if (mine and clear > 0) else GREY
        xo = 540 + (k - 2) * 110 + 10 * math.sin(t * 0.9 + k * 1.7)
        for j in range(-3, 4):
            seg = 78 * math.sqrt(max(0, 1 - (j / 3.6) ** 2))
            jit = 8 * math.sin(t * 2.1 + j * 1.3 + k)
            stroke(d, [(xo - seg + jit, y + j * 9), (xo + seg + jit, y + j * 9)], col, ra, 5)


def hanger(d, x, y, a):
    stroke(d, [(x, y - 70), (x, y - 92)], INK, a, 3)
    d.arc(P(x - 14, y - 120, x + 14, y - 92), 0, 200, fill=C(INK, a), width=3 * S)
    stroke(d, [(x - 110, y), (x, y - 70), (x + 110, y), (x - 110, y)], INK, a, 3)
    # 衣裳
    stroke(d, [(x - 95, y + 4), (x - 140, y + 70), (x - 105, y + 92), (x - 80, y + 60), (x - 80, y + 230),
               (x + 80, y + 230), (x + 80, y + 60), (x + 105, y + 92), (x + 140, y + 70), (x + 95, y + 4)],
           INK, a, 3)


def chat(d, x, y, a, erase):
    d.rounded_rectangle(P(x - 150, y - 90, x + 150, y + 110), radius=28 * S, outline=C(INK, a), width=3 * S)
    d.polygon([tuple(P(x - 90, y + 108)), tuple(P(x - 120, y + 150)), tuple(P(x - 50, y + 108))],
              fill=C(PAPER, 1), outline=None)
    stroke(d, [(x - 90, y + 110), (x - 120, y + 150), (x - 50, y + 110)], INK, a, 3)
    stroke(d, [(x - 105, y - 40), (x + 95, y - 40)], GREY, a, 6)
    stroke(d, [(x - 105, y + 10), (x + 60, y + 10)], GREY, a, 6)
    e = 1 - erase
    if e > 0.02:
        stroke(d, [(x - 105, y + 60), (x - 105 + 180 * e, y + 60)], GREY, a, 6)
    if erase < 1:   # 光标
        cx = x - 105 + 180 * e + 10
        if int(t_global * 2) % 2 == 0:
            stroke(d, [(cx, y + 42), (cx, y + 78)], INK, a, 3)


t_global = 0.0


def sc_daily(d, t, ts):
    dim = 1 - 0.65 * fade(t, ts[2], 1.0)
    hanger(d, 300, 560, fade(t, ts[0]) * dim)
    chat(d, 760, 620, fade(t, ts[1]) * dim, fade(t, ts[1] + 1.0, 1.2))
    a = fade(t, ts[2] + 0.2, 1.2)
    stroke(d, [(300, 800), (520, 960)], GREY, a * 0.6, 2)
    stroke(d, [(760, 780), (560, 960)], GREY, a * 0.6, 2)
    text(d, "以为", 540, 1040, 120, INK, a, "bold")


def sc_flag(d, t, ts):
    still = fade(t, ts[1], 1.8)                    # “仁者心动”：风止幡静
    a = fade(t, 0, 1.0)
    px = 330
    # 风（先画，位于幡与旗杆之后）
    wind = a * (1 - still)
    for k in range(4):
        ph = (t * 0.35 + k * 0.27) % 1.0
        x0 = -200 + 1400 * ph
        y0 = 470 + k * 120
        pts = [(x0 + s, y0 + 18 * math.sin(s / 60 + k)) for s in range(0, 260, 10)]
        stroke(d, pts, GREY, wind * 0.7 * math.sin(math.pi * ph), 3)
    stroke(d, [(px, 360), (px, 1110)], INK, a, 7)
    stroke(d, [(px - 10, 380), (px + 150, 380)], INK, a, 6)
    amp = 46 * (1 - still)
    left, right = [], []
    for i in range(41):
        u = i / 40
        y = 390 + 470 * u
        dx = amp * u ** 1.3 * math.sin(2 * math.pi * (u * 1.4) - t * 3.2)
        left.append((px + 20 + dx, y))
        right.append((px + 140 + dx + 0.4 * amp * u * math.sin(-t * 3.2 + 1), y))
    poly = left + right[::-1]
    d.polygon([tuple(P(*p)) for p in poly], fill=C(FAINT, 0.6 * a), outline=None)
    stroke(d, poly + [poly[0]], INK, a, 3)
    # 心
    enso(d, 760, 700, 125, (t - ts[1] - 0.3) / 1.6, INK, 1.0, 16, seed=5)
    text(d, "心", 760, 700, 120, INK, fade(t, ts[1] + 1.0, 1.0), "bold")


_dust = None


def sc_mirror(d, t, ts):
    global _dust
    cx, cy, r = 540, 720, 245
    enso(d, cx, cy, r, 1.0, INK, fade(t, 0, 1.0), 30, seed=8)
    if _dust is None:
        rng = np.random.default_rng(11)
        ang, rad = rng.uniform(0, 2 * math.pi, 150), r * 0.82 * np.sqrt(rng.uniform(0, 1, 150))
        _dust = [(cx + rr * math.cos(q), cy + rr * math.sin(q), rng.uniform(1.5, 7), rng.uniform(0.3, 0.8))
                 for q, rr in zip(ang, rad)]
    sweep = -200 + 1500 * ease((t - ts[1] - 0.1) / 1.6)       # 拂拭位置
    a0 = fade(t, ts[0] - 0.2, 1.0)
    for x, y, s, al in _dust:
        if x + 0.35 * (y - cy) > sweep:
            d.ellipse(P(x - s, y - s, x + s, y + s), fill=C(INK, a0 * al))
    if 0 < sweep < 1300:                                       # 拂尘的一道淡墨
        for k in range(10):
            xs = sweep - 30 - k * 9
            stroke(d, [(xs + 0.35 * 300, cy - 300), (xs - 0.35 * 300, cy + 300)], FAINT, 0.5 * (1 - k / 10), 6)
    clean = fade(t, ts[2], 1.5)
    if clean > 0:
        d.ellipse(P(cx - 92, cy - 92, cx + 92, cy + 92), fill=C(FAINT, 0.55 * clean),
                  outline=C(INK, clean), width=3 * S)
    seal(d, "照见", 880, 1470, 40, fade(t, ts[3] + 1.0, 0.5))


DRAW = {"enso": sc_enso, "moon": sc_moon, "daily": sc_daily, "flag": sc_flag, "mirror": sc_mirror}


def wrap(s, size, maxw):
    f = F("serif", size)
    if f.getlength(s) / S <= maxw:
        return [s]
    mid = len(s) / 2
    cands = [i + 1 for i, ch in enumerate(s[:-1]) if ch in "，、：；"]
    best = min(cands, key=lambda i: abs(i - mid), default=None)
    k = best if best is not None and abs(best - mid) <= len(s) * 0.25 else round(mid)
    return [s[:k], s[k:]]


def captions(d, sc, t):
    """文案：按句出现；keep 的句子与下一句上下并列。"""
    lines = sc["lines"]
    groups, cur = [], []
    for ln in lines:
        cur.append(ln)
        if not ln.get("keep"):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    for gi, g in enumerate(groups):
        g_end = groups[gi + 1][0]["t0"] if gi + 1 < len(groups) else sc["end"] + 10
        if not (g[0]["t0"] - 0.1 <= t < g_end + 0.1):
            continue
        out = 1 - fade(t, g_end - 0.45, 0.45)
        final = g[-1].get("final")
        size = 72 if final else 60
        rows = []
        for ln in g:
            for r in wrap(ln["show"], size, 820):
                rows.append((r, ln))
        step = size * 1.55
        y0 = TEXT_Y - (len(rows) - 1) * step / 2
        for i, (r, ln) in enumerate(rows):
            a = fade(t, ln["t0"] - 0.1, 0.7) * out
            text(d, r, 540, y0 + i * step, size, INK, a, "bold" if final else "serif")
        note = next((ln for ln in g if ln.get("note")), None)
        if note:
            text(d, note["note"], 540, y0 + len(rows) * step + 10, 30, GREY,
                 fade(t, note["t1"], 0.8) * out, "light")


# ---------------------------------------------------------------- 逐帧渲染
_bg = None
TL = None


def background():
    """宣纸质感：低频纤维噪声 + 暗角。"""
    rng = np.random.default_rng(3)
    small = rng.normal(0, 1, (H // 4, W // 4)).astype(np.float32)
    n1 = np.array(Image.fromarray(((small * 18) + 128).clip(0, 255).astype(np.uint8))
                  .resize((W * S, H * S), Image.BICUBIC)
                  .filter(ImageFilter.GaussianBlur(6)), dtype=np.float32) - 128
    fine = rng.normal(0, 3.0, (H * S, W * S)).astype(np.float32)
    yy, xx = np.mgrid[0:H * S, 0:W * S].astype(np.float32)
    vig = ((xx / (W * S) - 0.5) ** 2 + (yy / (H * S) - 0.5) ** 2) * 38
    base = np.array(PAPER, dtype=np.float32)[None, None, :]
    img = base + (n1 * 0.35 + fine - vig)[:, :, None] * np.array([1.0, 1.0, 1.05])[None, None, :]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def render(i):
    global GA, t_global
    t = t_global = i / FPS
    sc = next((s for s in TL if s["start"] <= t < s["end"]), TL[-1])
    lt = t - sc["start"]
    img = _bg.copy()
    d = ImageDraw.Draw(img)
    last = sc is TL[-1]
    first = sc is TL[0]
    GA = min(1.0 if first else fade(lt, 0, 0.5), 1.0 if last else 1 - fade(t, sc["end"] - 0.5, 0.5))
    ts = [ln["t0"] - sc["start"] for ln in sc["lines"]]
    DRAW[sc["key"]](d, lt, ts)
    GA = 1.0
    captions(d, sc, t)
    return img.reduce(S).tobytes()


def render_cover(path):
    """封面：圆相 + 标题。"""
    global GA
    GA = 1.0
    img = _bg.copy()
    d = ImageDraw.Draw(img)
    enso(d, 540, 760, 300, 1.0, INK, 1.0, 40, seed=1)
    vtext(d, "照见", 540, 660, 150, INK, 1.0, "bold", 1.3)
    text(d, "你在意的，", 540, 1300, 72, INK, 1.0, "serif")
    text(d, "从来不是别人的眼光", 540, 1410, 72, INK, 1.0, "serif")
    seal(d, "禅心", 870, 1560, 34, 1.0)
    img.reduce(S).save(path)


def init_worker(args, tl):
    global ARGS, TL, _bg
    ARGS, TL = args, tl
    _bg = background()


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--assets", default=os.path.join(here, "assets"))
    ap.add_argument("--out", default=os.path.join(here, "zhaojian.mp4"))
    ap.add_argument("--cover", default=None, help="同时导出封面 PNG")
    ap.add_argument("--preview", type=float, default=None, help="仅导出某一时刻的单帧 PNG")
    ap.add_argument("--voice", default=os.environ.get("MINIMAX_VOICE", "Chinese (Mandarin)_Radio_Host"))
    ap.add_argument("--model", default=os.environ.get("MINIMAX_MODEL", "speech-2.6-hd"))
    ap.add_argument("--speed", type=float, default=0.82)
    ARGS = ap.parse_args()

    voice = synth_lines(ARGS)
    tl, total = build_timeline(voice)
    print(f"总时长 {total:.1f} s")
    for s in tl:
        print(f"  {s['key']:7s} {s['start']:6.2f} – {s['end']:6.2f}")
    light = [dict(s, lines=[{k: v for k, v in l.items() if k != "audio"} for l in s["lines"]]) for s in tl]

    if ARGS.preview is not None:
        init_worker(ARGS, light)
        Image.frombytes("RGB", (W, H), render(int(ARGS.preview * FPS))).save(
            os.path.splitext(ARGS.out)[0] + f"_{ARGS.preview:.1f}s.png")
        return

    wav = os.path.splitext(ARGS.out)[0] + ".wav"
    make_audio(tl, total, wav)
    nframes = int(total * FPS)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-af", "loudnorm=I=-15:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
         "-movflags", "+faststart", "-shortest", ARGS.out], stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init_worker, initargs=(ARGS, light)) as pool:
        for k, buf in enumerate(pool.imap(render, range(nframes), chunksize=8)):
            ff.stdin.write(buf)
            if k % 300 == 0:
                print(f"  帧 {k}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    if ARGS.cover:
        init_worker(ARGS, light)
        render_cover(ARGS.cover)
    print("完成：", ARGS.out)


if __name__ == "__main__":
    main()
