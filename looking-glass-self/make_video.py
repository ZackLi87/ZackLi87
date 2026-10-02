"""
生成竖屏短视频：《“我眼中的别人眼中的我”的证明》

流程：离线语音合成（sherpa-onnx + Matcha 中文模型）→ 按语音时长排布时间轴
→ Pillow 逐帧绘制（2 倍超采样）→ 合成背景音乐 → ffmpeg 封装 H.264/AAC。

用法：python3 make_video.py [--assets DIR] [--out OUT.mp4] [--preview SECONDS]
资源（字体、语音模型）由 fetch_assets.sh 下载到 DIR（默认 ./assets）。
"""
import argparse
import hashlib
import math
import os
import subprocess
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, S = 1080, 1920, 30, 2          # 输出分辨率、帧率、超采样倍数
SR = 44100                                # 混音采样率
SPEED = 1.22                              # 语速

BG = (14, 17, 24)
TEXT = (238, 234, 226)
MUTED = (128, 136, 150)
DIM = (62, 68, 80)
AMBER = (245, 178, 66)     # 真实的我 X
CYAN = (88, 198, 216)      # 别人眼中的我 f(X)
VIOLET = (172, 142, 245)   # 我眼中的别人眼中的我 g(f(X))
RED = (226, 96, 104)

ARGS = None
GA = 1.0                   # 当前帧的全局不透明度（用于转场）

# ---------------------------------------------------------------- 文案与分镜
# 每个场景：step（顶部进度标识）、lines（字幕/旁白），旁白文本可用 say 覆盖。
SCENES = [
    dict(key="hook", step=None, lines=[
        "有一句话，听起来很绕：",
        "别人眼中的自己，不是真的自己；",
        "我眼中的别人眼中的自己，才是真的自己。",
        "下面分三步，给出它的证明。",
    ]),
    dict(key="define", step=0, lines=[
        ("先作定义：真实的我，记作 X。", "先作定义：真实的我，记作艾克斯。"),
        "别人看我，要经过他的经历、偏好和情绪的过滤，",
        "得到的像，就是“别人眼中的我”。",
    ]),
    dict(key="step1", step=1, lines=[
        "第一步。同一个我，在一百个人眼中，",
        "会呈现一百种不同的形象。",
        "这些形象彼此矛盾，不可能都等于真实的我。",
        "所以，别人眼中的自己，不是真的自己。",
    ]),
    dict(key="step2", step=2, lines=[
        "第二步。别人眼中的我，存在于别人的头脑里，",
        "我无法直接看到。",
        "我能看到的，只是我对它的推测，",
        "也就是：我眼中的别人眼中的我。",
    ]),
    dict(key="step3", step=3, lines=[
        "第三步。回想一下：",
        "出门前换掉的那件衣服，",
        "发送前删掉的那句话，",
        "会议上没有说出口的那个想法。",
        "决定这些的，并不是别人真实的看法，",
        "而是“我以为别人会怎么看”。",
    ]),
    dict(key="qed", step=4, lines=[
        "一个人的言行、选择乃至情绪，都由这个“以为”决定。",
        "因此，真正塑造“我”的，是我眼中的别人眼中的我。",
        "证毕。",
    ]),
    dict(key="theory", step=None, lines=[
        "这并不只是一个段子。",
        ("1902年，美国社会学家库利提出“镜中自我”理论：",
         "一九零二年，美国社会学家库利，提出镜中自我理论："),
        "人通过想象他人如何看待自己，来形成对自我的认知。",
    ]),
    dict(key="mirror", step=None, lines=[
        "所以，与其反复揣测别人怎么看你，",
        "不如检查一下你心里的那面镜子：",
        "它是否放大了缺点，又忽略了优点？",
        "镜子是自己挂上的，也能由自己擦亮。",
        "你心里的那面镜子，照出的是怎样的你？",
    ]),
]
STEPS = ["定义", "第一步", "第二步", "第三步", "证毕"]
LEAD, GAP, TAIL, FINAL_HOLD = 0.35, 0.24, 0.6, 3.0


# ---------------------------------------------------------------- 语音合成
def tts_engine(assets):
    import sherpa_onnx
    d = os.path.join(assets, "matcha-icefall-zh-baker")
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=f"{d}/model-steps-3.onnx",
                vocoder=os.path.join(assets, "vocos-22khz-univ.onnx"),
                lexicon=f"{d}/lexicon.txt", tokens=f"{d}/tokens.txt",
                dict_dir=f"{d}/dict"),
            num_threads=4),
        rule_fsts=f"{d}/phone.fst,{d}/date.fst,{d}/number.fst")
    return sherpa_onnx.OfflineTts(cfg)


def synth_lines(assets, cache):
    """为每一行旁白合成语音，返回 [(scene_idx, line_idx, show, samples@SR)]。"""
    os.makedirs(cache, exist_ok=True)
    eng, out = None, []
    for si, sc in enumerate(SCENES):
        for li, ln in enumerate(sc["lines"]):
            show, say = (ln if isinstance(ln, tuple) else (ln, ln))
            path = os.path.join(cache, hashlib.md5(f"{say}|{SPEED}".encode()).hexdigest() + ".npy")
            if not os.path.exists(path):
                eng = eng or tts_engine(assets)
                a = eng.generate(say, sid=0, speed=SPEED)
                x = np.array(a.samples, dtype=np.float32)
                # 线性插值重采样到 SR，并裁去首尾静音
                n = int(len(x) * SR / a.sample_rate)
                x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
                nz = np.where(np.abs(x) > 0.01)[0]
                x = x[max(nz[0] - 800, 0): nz[-1] + 2000] if len(nz) else x
                np.save(path, x.astype(np.float32))
            out.append((si, li, show, np.load(path)))
    return out


def build_timeline(voice):
    """计算每个场景起止时间与每行字幕的起止时间（秒）。"""
    scenes, t = [], 0.0
    for si, sc in enumerate(SCENES):
        start, cur, lines = t, t + LEAD, []
        for (vsi, li, show, x) in voice:
            if vsi != si:
                continue
            d = len(x) / SR
            lines.append(dict(show=show, t0=cur, t1=cur + d, audio=x))
            cur += d + GAP
        end = cur - GAP + (FINAL_HOLD if si == len(SCENES) - 1 else TAIL)
        scenes.append(dict(sc, start=start, end=end, lines=lines))
        t = end
    return scenes, t


# ---------------------------------------------------------------- 背景音乐
def make_music(total):
    n = int(total * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, dtype=np.float64)
    hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]   # Am F C G
    bar = 4.0
    for k in range(int(total / bar) + 2):
        c = chords[k % 4]
        s0, s1 = int(k * bar * SR), min(int((k + 1) * bar * SR + 1.5 * SR), n)
        if s0 >= n:
            break
        tt = t[s0:s1] - k * bar
        env = np.minimum(tt / 1.2, 1.0) * np.exp(-np.maximum(tt - bar, 0) / 0.6)
        seg = np.zeros_like(tt)
        for m in c + [c[0] + 12]:
            f = hz(m)
            seg += np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt + 0.3)
        seg += 0.8 * np.sin(2 * np.pi * hz(c[0] - 12) * tt)              # 低音
        out[s0:s1] += seg * env
        # 每小节两个轻柔的拨弦音
        for j, m in enumerate([c[2] + 12, c[1] + 12]):
            p0 = int((k * bar + j * 2.0) * SR)
            if p0 < n:
                pt = np.arange(min(int(1.6 * SR), n - p0)) / SR
                out[p0:p0 + len(pt)] += 0.5 * np.sin(2 * np.pi * hz(m) * pt) * np.exp(-pt / 0.35)
    out /= np.max(np.abs(out)) + 1e-9
    fade = np.minimum(1, np.minimum(t / 1.5, (total - t) / 2.5))
    return out * fade * 0.075


def make_audio(scenes, total, path):
    n = int(total * SR) + SR
    voice = np.zeros(n)
    for sc in scenes:
        for ln in sc["lines"]:
            s = int(ln["t0"] * SR)
            voice[s:s + len(ln["audio"])] += ln["audio"]
    voice *= 0.89 / (np.max(np.abs(voice)) + 1e-9)
    mix = voice + make_music(n / SR)
    mix = np.clip(mix, -1, 1)
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
        name = {"sans": "NotoSansSC-Regular.otf", "bold": "NotoSansSC-Bold.otf",
                "serif": "NotoSerifSC-Bold.otf"}[kind]
        _fonts[key] = ImageFont.truetype(os.path.join(ARGS.assets, "fonts", name), size * S)
    return _fonts[key]


def C(col, a=1.0):
    a = max(0.0, min(1.0, a * GA))
    return tuple(int(BG[i] + (col[i] - BG[i]) * a) for i in range(3))


def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def fade(t, t0, dur=0.45):
    return ease((t - t0) / dur)


def P(*v):
    return [int(round(u * S)) for u in v]


def text(d, s, x, y, size, col, a=1.0, kind="bold", anchor="mm"):
    if a <= 0.01:
        return
    d.text(P(x, y), s, font=F(kind, size), fill=C(col, a), anchor=anchor)


def rich(d, segs, cx, y, size, a=1.0, kind="bold"):
    """居中绘制多色文本。segs: [(文本, 颜色)] 或 [(文本, 颜色, 'sub')]。"""
    if a <= 0.01:
        return
    parts = []
    for sg in segs:
        sub = len(sg) > 2
        f = F(kind, int(size * 0.55) if sub else size)
        parts.append((sg[0], sg[1], f, sub, f.getlength(sg[0]) / S))
    x = cx - sum(p[4] for p in parts) / 2
    for s, col, f, sub, w in parts:
        yy = y + (size * 0.28 if sub else 0)
        d.text(P(x, yy), s, font=f, fill=C(col, a), anchor="lm")
        x += w


def circle(d, x, y, r, col, a=1.0, width=4, fill=None):
    if a <= 0.01:
        return
    d.ellipse(P(x - r, y - r, x + r, y + r), outline=C(col, a), width=width * S,
              fill=C(fill[0], fill[1] * a) if fill else None)


def rrect(d, x0, y0, x1, y1, col, a=1.0, width=3, fill=None, r=18):
    if a <= 0.01:
        return
    d.rounded_rectangle(P(x0, y0, x1, y1), radius=r * S, outline=C(col, a) if col else None,
                        width=width * S, fill=C(fill[0], fill[1] * a) if fill else None)


def arrow(d, x0, y0, x1, y1, col, a=1.0, prog=1.0, width=4, dashed=False, head=18):
    if a <= 0.01 or prog <= 0:
        return
    prog = ease(prog)
    xe, ye = x0 + (x1 - x0) * prog, y0 + (y1 - y0) * prog
    if dashed:
        L = math.hypot(xe - x0, ye - y0)
        n = int(L // 22)
        for i in range(n + 1):
            u0, u1 = i * 22 / max(L, 1), min((i * 22 + 12) / max(L, 1), 1)
            if u0 >= 1:
                break
            d.line(P(x0 + (xe - x0) * u0, y0 + (ye - y0) * u0, x0 + (xe - x0) * u1,
                     y0 + (ye - y0) * u1), fill=C(col, a), width=width * S)
    else:
        d.line(P(x0, y0, xe, ye), fill=C(col, a), width=width * S)
    if prog > 0.85:
        ang = math.atan2(y1 - y0, x1 - x0)
        pts = [(xe, ye), (xe - head * math.cos(ang - 0.45), ye - head * math.sin(ang - 0.45)),
               (xe - head * math.cos(ang + 0.45), ye - head * math.sin(ang + 0.45))]
        d.polygon([tuple(P(*p)) for p in pts], fill=C(col, a))


def blob(d, x, y, r, col, a=1.0, seed=0, amp=0.16, t=0.0, width=4, fill=0.14):
    """不规则的“像”：表示经过过滤、发生畸变的形象。"""
    if a <= 0.01:
        return
    rng = np.random.default_rng(seed)
    ks, ph = rng.integers(2, 6, 3), rng.uniform(0, 6.28, 3)
    pts = []
    for i in range(96):
        th = 2 * math.pi * i / 96
        rr = r * (1 + amp * sum(math.sin(k * th + p + 0.6 * t) for k, p in zip(ks, ph)) / 2)
        pts.append(tuple(P(x + rr * math.cos(th), y + rr * math.sin(th))))
    d.polygon(pts, fill=C(col, fill * a))
    d.line(pts + [pts[0]], fill=C(col, a), width=width * S, joint="curve")


def person(d, x, y, r, col, a=1.0, width=4):
    """简笔人像：头部 + 肩部。(x, y) 为头部中心。"""
    if a <= 0.01:
        return
    circle(d, x, y, r, col, a, width)
    d.arc(P(x - 1.7 * r, y + 1.25 * r, x + 1.7 * r, y + 4.4 * r), 180, 360,
          fill=C(col, a), width=width * S)


def bubble(d, x, y, rx, ry, col, a=1.0, width=3, fill=None, tail=None):
    if a <= 0.01:
        return
    d.ellipse(P(x - rx, y - ry, x + rx, y + ry), outline=C(col, a), width=width * S,
              fill=C(fill[0], fill[1] * a) if fill else None)
    if tail:
        for i, (fx, rr) in enumerate([(0.35, 12), (0.7, 7)]):
            cx, cy = x + (tail[0] - x) * fx, y + ry + (tail[1] - y - ry) * fx
            circle(d, cx, cy, rr, col, a, width)


def tag(d, s, x, y, size, col, a=1.0, fill=0.12):
    if a <= 0.01:
        return
    w = F("bold", size).getlength(s) / S
    rrect(d, x - w / 2 - 22, y - size * 0.85, x + w / 2 + 22, y + size * 0.85, col, a, 3,
          fill=(col, fill), r=int(size * 0.85))
    text(d, s, x, y, size, col, a)


def conclusion(d, label, body, y, a, col=AMBER):
    """引理/结论框。"""
    if a <= 0.01:
        return
    dy = (1 - a) * 30
    rrect(d, 70, y - 62 + dy, 1010, y + 62 + dy, col, a, 3, fill=(col, 0.08), r=22)
    text(d, label, 110, y + dy, 38, col, a, "bold", "lm")
    text(d, body, 110 + F("bold", 38).getlength(label) / S + 18, y + dy, 38, TEXT, a, "bold", "lm")


def wrap(s, size, maxw):
    """字幕换行：需要两行时，优先在靠近中点的标点后断开，使两行长度均衡。"""
    f = F("bold", size)
    if f.getlength(s) / S <= maxw:
        return [s]
    mid = len(s) / 2
    cands = [i + 1 for i, ch in enumerate(s[:-1]) if ch in "，、：；"]
    best = min(cands, key=lambda i: abs(i - mid), default=None)
    k = best if best is not None and abs(best - mid) <= len(s) * 0.2 else round(mid)
    return [s[:k], s[k:]]


# ---------------------------------------------------------------- 各场景画面
def sc_hook(d, t, ts):
    on = lambda i: 1.0 if (ts[i] <= t < (ts[i + 1] if i + 1 < len(ts) else 1e9)) else 0.55
    tag(d, "命　题", 540, 430, 34, AMBER, fade(t, 0, 0.3))
    a1 = fade(t, 0, 0.3) * (1.0 if t < ts[1] else on(1))
    rich(d, [("别人眼中的", CYAN), ("自己", AMBER)], 540, 580, 74, a1, "serif")
    rich(d, [("不是", TEXT), ("真的自己", TEXT)], 540, 680, 74, a1, "serif")
    a2 = fade(t, 0, 0.3) * (1.0 if t < ts[1] else on(2))
    rich(d, [("我眼中的", VIOLET), ("别人眼中的", CYAN), ("自己", AMBER)], 540, 850, 74, a2, "serif")
    rich(d, [("才是", TEXT), ("真的自己", TEXT)], 540, 950, 74, a2, "serif")
    for i in range(3):
        a = fade(t, ts[3] + 0.25 * i)
        circle(d, 360 + 180 * i, 1150, 46, AMBER, a, 3, fill=(AMBER, 0.12))
        text(d, str(i + 1), 360 + 180 * i, 1148, 46, AMBER, a, "serif")
    text(d, "三步证明", 540, 1245, 34, MUTED, fade(t, ts[3] + 0.6))


def sc_define(d, t, ts):
    a = fade(t, ts[0])
    circle(d, 230, 720, 112, AMBER, a, 5, fill=(AMBER, 0.12))
    text(d, "X", 230, 712, 100, AMBER, a, "serif")
    text(d, "真实的我", 230, 880, 38, TEXT, a)
    # 过滤器
    a = fade(t, ts[1])
    arrow(d, 352, 720, 470, 720, MUTED, a, (t - ts[1]) / 0.5)
    rrect(d, 490, 560, 590, 880, CYAN, a * 0.8, 3, fill=(CYAN, 0.06), r=46)
    for i in range(5):
        yy = 600 + i * 60
        d.line(P(510, yy + 20, 570, yy - 20), fill=C(CYAN, 0.35 * a), width=3 * S)
    text(d, "过滤", 540, 520, 36, CYAN, a)
    for i, w in enumerate(["经历", "偏好", "情绪"]):
        tag(d, w, 540, 950 + i * 78, 30, MUTED, fade(t, ts[1] + 0.9 + 0.55 * i), 0.06)
    # 像
    a = fade(t, ts[2])
    arrow(d, 610, 720, 728, 720, MUTED, a, (t - ts[2]) / 0.5)
    blob(d, 860, 720, 112, CYAN, a, seed=3, t=t, width=5)
    rich(d, [("f", CYAN), ("(", CYAN), ("X", AMBER), (")", CYAN)], 860, 712, 76, a, "serif")
    text(d, "别人眼中的我", 860, 880, 38, TEXT, a)
    text(d, "f：别人的观察（含过滤）", 540, 1220, 32, MUTED, fade(t, ts[2] + 0.8), "sans")


LABELS = ["高冷", "热情", "靠谱", "敷衍", "有趣", "无聊"]


def sc_step1(d, t, ts):
    cx, cy = 540, 720
    circle(d, cx, cy, 82, AMBER, fade(t, ts[0]), 5, fill=(AMBER, 0.12))
    text(d, "X", cx, cy - 6, 80, AMBER, fade(t, ts[0]), "serif")
    for i in range(6):
        ang = -math.pi / 2 + i * math.pi / 3
        px, py = cx + 215 * math.cos(ang), cy + 215 * math.sin(ang)
        a = fade(t, ts[0] + 0.5 + 0.12 * i)
        person(d, px, py - 20, 18, CYAN, a * 0.85, 3)
        a2 = fade(t, ts[1] + 0.15 * i)
        bx, by = cx + 375 * math.cos(ang), cy + 335 * math.sin(ang)
        blob(d, bx, by, 62, CYAN, a2, seed=10 + i, amp=0.22, t=t, width=3, fill=0.1)
        text(d, LABELS[i], bx, by - 2, 34, TEXT, a2)
    a3 = fade(t, ts[2]) * (1 - fade(t, ts[3], 0.3))
    rich(d, [("f", CYAN), ("1", CYAN, "sub"), ("(X)", CYAN), ("  ≠  ", TEXT), ("f", CYAN),
             ("2", CYAN, "sub"), ("(X)", CYAN), ("  ≠ … ≠  ", TEXT), ("X", AMBER)],
         540, 1235, 54, a3, "serif")
    conclusion(d, "引理一", "别人眼中的我 ≠ 真实的我", 1235, fade(t, ts[3] + 0.2))


def sc_step2(d, t, ts):
    # 他人
    a = fade(t, ts[0])
    person(d, 800, 760, 62, CYAN, a, 5)
    text(d, "别人", 800, 1010, 36, CYAN, a)
    bubble(d, 790, 470, 175, 115, CYAN, a, 3, fill=(CYAN, 0.06), tail=(800, 680))
    hide = fade(t, ts[1], 0.6)
    blob(d, 740, 470, 48, CYAN, a * (1 - hide), seed=3, t=t, width=3)
    rich(d, [("f", CYAN), ("(", CYAN), ("X", AMBER), (")", CYAN)], 850, 470, 50, a * (1 - hide), "serif")
    if hide > 0:
        d.ellipse(P(790 - 175, 470 - 115, 790 + 175, 470 + 115), fill=C(DIM, hide * 0.9))
        text(d, "？", 790, 445, 100, TEXT, hide, "serif")
        text(d, "无法直接观测", 790, 530, 28, MUTED, hide)
    # 我
    a = fade(t, ts[0] + 0.3)
    person(d, 280, 760, 62, AMBER, a, 5)
    text(d, "我", 280, 1010, 36, AMBER, a)
    a = fade(t, ts[2])
    bubble(d, 270, 470, 175, 115, VIOLET, a, 3, fill=(VIOLET, 0.08), tail=(280, 680))
    rich(d, [("g", VIOLET), ("(", VIOLET), ("f", CYAN), ("(", CYAN), ("X", AMBER), (")", CYAN),
             (")", VIOLET)], 270, 470, 54, a, "serif")
    arrow(d, 455, 470, 600, 470, VIOLET, a, (t - ts[2]) / 0.6, 3, dashed=True)
    text(d, "推测", 528, 432, 30, VIOLET, fade(t, ts[2] + 0.5))
    a = fade(t, ts[3])
    rich(d, [("g", VIOLET), ("：我对别人看法的推测", TEXT)], 540, 1110, 36, a, "sans")
    conclusion(d, "引理二", "我能接触的，只有 g(f(X))", 1215, fade(t, ts[3] + 1.0), VIOLET)


def sc_step3(d, t, ts):
    items = ["出门前换掉的衣服", "发送前删掉的句子", "会上没说出口的想法"]
    ys = [440, 600, 760]
    for i, s in enumerate(items):
        a = fade(t, ts[i + 1])
        dx = (1 - a) * -40
        rrect(d, 70 + dx, ys[i] - 55, 600 + dx, ys[i] + 55, TEXT, a * 0.5, 2, fill=(TEXT, 0.05))
        text(d, s, 335 + dx, ys[i], 38, TEXT, a)
    # 候选决定因素
    a4 = fade(t, ts[4])
    rrect(d, 730, 395, 1000, 505, CYAN, a4, 3, fill=(CYAN, 0.08))
    rich(d, [("f", CYAN), ("(", CYAN), ("X", AMBER), (")", CYAN)], 865, 440, 50, a4, "serif")
    text(d, "别人真实的看法", 865, 485, 22, MUTED, a4, "sans")
    for i in range(3):
        arrow(d, 612, ys[i], 718, 450, MUTED, a4 * 0.6, (t - ts[4] - 0.1 * i) / 0.5, 3, dashed=True)
    x_ = fade(t, ts[4] + 1.0, 0.3)
    if x_ > 0:
        d.line(P(745, 410, 745 + 240 * x_, 410 + 80 * x_), fill=C(RED, x_), width=5 * S)
        d.line(P(985, 410, 985 - 240 * x_, 410 + 80 * x_), fill=C(RED, x_), width=5 * S)
    a5 = fade(t, ts[5])
    rrect(d, 730, 655, 1000, 785, VIOLET, a5, 4, fill=(VIOLET, 0.14))
    rich(d, [("g", VIOLET), ("(", VIOLET), ("f", CYAN), ("(", CYAN), ("X", AMBER), (")", CYAN),
             (")", VIOLET)], 865, 705, 48, a5, "serif")
    text(d, "我以为的看法", 865, 757, 24, VIOLET, a5, "sans")
    for i in range(3):
        arrow(d, 612, ys[i], 718, 720, VIOLET, a5, (t - ts[5] - 0.12 * i) / 0.5, 4)
    text(d, "决定因素", 865, 830, 30, VIOLET, fade(t, ts[5] + 0.7))
    conclusion(d, "引理三", "决定行为的是 g(f(X))，而非 f(X)", 1060,
               fade(t, ts[5] + 1.2), VIOLET)


def sc_qed(d, t, ts):
    rich(d, [("∵  言行、选择、情绪  ←  ", TEXT), ("g", VIOLET), ("(", VIOLET), ("f", CYAN),
             ("(", CYAN), ("X", AMBER), (")", CYAN), (")", VIOLET)], 540, 450, 46, fade(t, ts[0]), "bold")
    rich(d, [("∴  塑造“我”的  =  ", TEXT), ("g", VIOLET), ("(", VIOLET), ("f", CYAN),
             ("(", CYAN), ("X", AMBER), (")", CYAN), (")", VIOLET)], 540, 560, 46, fade(t, ts[1]), "bold")
    a = fade(t, ts[1] + 1.2, 0.6)
    rrect(d, 80, 680, 1000, 960, AMBER, a * 0.7, 3, fill=(AMBER, 0.06), r=28)
    rich(d, [("我眼中的", VIOLET), ("别人眼中的", CYAN), ("我", AMBER)], 540, 765, 66, a, "serif")
    rich(d, [("=  真正的我", TEXT)], 540, 875, 70, a, "serif")
    a = fade(t, ts[2], 0.35)
    sc = 1 + 0.4 * (1 - a)
    d.rectangle(P(395 - 22 * sc, 1110 - 22 * sc, 395 + 22 * sc, 1110 + 22 * sc), fill=C(AMBER, a))
    text(d, "证毕", 560, 1106, 90 if a >= 1 else int(90 * sc), AMBER, a, "serif")


def sc_theory(d, t, ts):
    tag(d, "理论依据", 540, 420, 34, AMBER, fade(t, ts[0]))
    a = fade(t, ts[1])
    text(d, "镜中自我", 540, 560, 104, TEXT, a, "serif")
    text(d, "Looking-glass Self", 540, 668, 40, MUTED, a, "sans")
    text(d, "C. H. Cooley, 1902", 540, 725, 32, MUTED, a, "sans")
    rows = ["想象自己在他人眼中的形象", "想象他人对这一形象的评价", "由此产生自豪或羞愧等自我感受"]
    cols = [AMBER, CYAN, VIOLET]
    for i, s in enumerate(rows):
        a = fade(t, ts[2] + 0.5 * i)
        y = 860 + i * 125
        rrect(d, 90, y - 50, 990, y + 50, cols[i], a * 0.7, 2, fill=(cols[i], 0.07))
        circle(d, 145, y, 26, cols[i], a, 3, fill=(cols[i], 0.2))
        text(d, str(i + 1), 145, y - 2, 30, cols[i], a, "serif")
        text(d, s, 195, y, 36, TEXT, a, "bold", "lm")


def sc_mirror(d, t, ts):
    cx, cy, rx, ry = 540, 690, 210, 290
    a = fade(t, ts[0])
    d.ellipse(P(cx - rx - 22, cy - ry - 22, cx + rx + 22, cy + ry + 22), outline=C(AMBER, a * 0.9),
              width=10 * S)
    d.ellipse(P(cx - rx, cy - ry, cx + rx, cy + ry), fill=C((40, 46, 58), a))
    d.line(P(cx, cy + ry + 22, cx, cy + ry + 70), fill=C(AMBER, a * 0.9), width=8 * S)
    d.line(P(cx - 90, cy + ry + 72, cx + 90, cy + ry + 72), fill=C(AMBER, a * 0.9), width=8 * S)
    clean = fade(t, ts[3] + 0.4, 1.4)
    a1 = fade(t, ts[1]) * (1 - clean)
    blob(d, cx, cy - 40, 120, VIOLET, a1, seed=7, amp=0.32, t=t * 1.5, width=4)
    # 放大的缺点 / 被忽略的优点
    a2 = fade(t, ts[2]) * (1 - clean)
    tag(d, "缺点 ×3", 205, 470, 34, RED, a2)
    tag(d, "优点 ×0.3", 880, 910, 26, MUTED, a2 * 0.8)
    if clean > 0:
        circle(d, cx, cy - 40, 120, AMBER, clean, 5, fill=(AMBER, 0.14))
        text(d, "X", cx, cy - 48, 110, AMBER, clean, "serif")
        for i in range(4):   # 擦亮后的高光
            ang = 0.6 + i * 1.6
            sx, sy = cx + 160 * math.cos(ang), cy - 40 + 200 * math.sin(ang)
            s = 14 * clean * (0.7 + 0.3 * math.sin(t * 4 + i))
            d.polygon([tuple(P(sx, sy - s)), tuple(P(sx + s * 0.3, sy)), tuple(P(sx, sy + s)),
                       tuple(P(sx - s * 0.3, sy))], fill=C(TEXT, clean))
    a = fade(t, ts[4], 0.6)
    text(d, "你心里的那面镜子，", 540, 1140, 58, TEXT, a, "serif")
    text(d, "照出的是怎样的你？", 540, 1230, 58, AMBER, a, "serif")


DRAW = {"hook": sc_hook, "define": sc_define, "step1": sc_step1, "step2": sc_step2,
        "step3": sc_step3, "qed": sc_qed, "theory": sc_theory, "mirror": sc_mirror}


# ---------------------------------------------------------------- 逐帧渲染
_bg = None
TL = None


def background():
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    for y in range(60, H, 60):
        for x in range(60, W, 60):
            d.ellipse(P(x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=(24, 28, 37))
    return img


def header(d, sc, t):
    global GA
    keep = GA
    GA = 1.0
    text(d, "一句绕口话的“严格证明”", 540, 165, 34, MUTED, 1.0, "serif")
    if sc["step"] is not None:
        xs = [190 + 175 * i for i in range(5)]
        for i, s in enumerate(STEPS):
            cur = i == sc["step"]
            col = AMBER if cur else (TEXT if i < sc["step"] else DIM)
            text(d, s, xs[i], 240, 30, col, 1.0, "bold")
            if cur:
                d.line(P(xs[i] - 34, 268, xs[i] + 34, 268), fill=C(AMBER), width=4 * S)
    GA = keep


def subtitle(d, sc, t):
    global GA
    keep = GA
    for ln in sc["lines"]:
        nxt = [l["t0"] for l in sc["lines"] if l["t0"] > ln["t0"]]
        t_end = (nxt[0] if nxt else sc["end"] - 0.15) - 0.02
        if ln["t0"] - 0.05 <= t < t_end:
            GA = min(fade(t, ln["t0"] - 0.05, 0.15), 1 - fade(t, t_end - 0.12, 0.12))
            rows = wrap(ln["show"], 50, 860)
            y0 = 1430 - (len(rows) - 1) * 36
            for i, r in enumerate(rows):
                text(d, r, 540, y0 + i * 72, 50, TEXT, 1.0, "bold")
    GA = keep


def render(i):
    global GA
    t = i / FPS
    sc = next((s for s in TL if s["start"] <= t < s["end"]), TL[-1])
    lt = t - sc["start"]
    img = _bg.copy()
    d = ImageDraw.Draw(img)
    header(d, sc, t)
    last = sc is TL[-1]
    GA = min(fade(lt, 0, 0.35), 1.0 if last else 1 - fade(t, sc["end"] - 0.35, 0.35))
    ts = [ln["t0"] - sc["start"] for ln in sc["lines"]]
    DRAW[sc["key"]](d, lt, ts)
    GA = 1.0
    subtitle(d, sc, t)
    return img.reduce(S).tobytes()


def init_worker(args, tl):
    global ARGS, TL, _bg
    ARGS, TL = args, tl
    _bg = background()


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--assets", default=os.path.join(here, "assets"))
    ap.add_argument("--out", default=os.path.join(here, "looking_glass_self.mp4"))
    ap.add_argument("--preview", type=float, default=None, help="仅导出某一时刻的单帧 PNG")
    ap.add_argument("--cover", default=None, help="同时导出封面 PNG 的路径")
    ARGS = ap.parse_args()

    voice = synth_lines(ARGS.assets, os.path.join(ARGS.assets, "tts_cache"))
    tl, total = build_timeline(voice)
    print(f"总时长 {total:.1f} s")
    for s in tl:
        print(f"  {s['key']:7s} {s['start']:6.2f} – {s['end']:6.2f}")
    light = [{k: v for k, v in s.items()} for s in tl]
    for s in light:
        s["lines"] = [{k: v for k, v in l.items() if k != "audio"} for l in s["lines"]]

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
         "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
         "-af", "loudnorm=I=-15:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-movflags", "+faststart",
         "-shortest", ARGS.out], stdin=subprocess.PIPE)
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
        Image.frombytes("RGB", (W, H), render(int(2.0 * FPS))).save(ARGS.cover)
    print("完成：", ARGS.out)


if __name__ == "__main__":
    main()
