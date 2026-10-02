"""
kidkit：“好奇心实验室”少儿科普系列的统一画面风格与渲染框架。

每一期只需提供一个 episode 模块，其中定义：
  NUMBER, TITLE                      期号与标题
  SCENES                             文案（与 vidkit 的格式相同）
  DRAW = {key: fn(d, t, ts)}         各场景的绘制函数
  BG = {key: fn(t, ts) -> Image}     （可选）自定义背景，缺省为计算得到的白天天空
  setup(tl)                          （可选）子进程初始化时的预计算，如模拟
  COVER_T                            （可选）封面取帧时刻
然后调用 kidkit.run(episode_module)。
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import optics  # noqa: E402
import vidkit as vk  # noqa: E402

W, H, FPS, S = 1080, 1920, 30, 2
SERIES = "好奇心实验室"
ARGS = None
GA = 1.0
TL = None
EP = None

# ---------------------------------------------------------------- 由物理计算得到的颜色
SKY = optics.sky_color(1.0)                                 # 正午天空
SKY_TOP = vk.lerp_col(SKY, (36, 92, 205), 0.45)
SKY_LOW = vk.lerp_col(SKY, (255, 255, 255), 0.55)
SUN_NOON = optics.sun_color(1.0)
DUSK_TOP = (44, 52, 118)

WHITE = (255, 255, 255)
NAVY = (24, 42, 88)
INK = (40, 48, 70)
SUNY = (255, 210, 80)
RED = optics.wavelength_rgb(680)
BLUE = (50, 120, 255)
VIOLET = optics.wavelength_rgb(410)
GRASS = (112, 196, 120)
PANEL = (20, 26, 46)

# ---------------------------------------------------------------- 绘图工具
_fonts = {}


def F(kind, size):
    key = (kind, size)
    if key not in _fonts:
        name = {"bold": "NotoSansSC-Bold.otf", "reg": "NotoSansSC-Regular.otf",
                "fun": "ZCOOLKuaiLe-Regular.ttf"}[kind]
        _fonts[key] = ImageFont.truetype(os.path.join(ARGS.assets, "fonts", name), int(size * S))
    return _fonts[key]


def A(col, a=1.0):
    return tuple(col[:3]) + (int(255 * max(0.0, min(1.0, a * GA))),)


def P(*v):
    return [int(round(u * S)) for u in v]


def text(d, s, x, y, size, col=WHITE, a=1.0, kind="bold", anchor="mm", stroke=0, scol=NAVY):
    if a <= 0.01 or size <= 1:
        return
    d.text(P(x, y), s, font=F(kind, size), fill=A(col, a), anchor=anchor,
           stroke_width=int(stroke * S), stroke_fill=A(scol, a) if stroke else None)


def circle(d, x, y, r, fill=None, a=1.0, outline=None, width=0):
    if a <= 0.01 or r <= 0:
        return
    d.ellipse(P(x - r, y - r, x + r, y + r), fill=A(fill, a) if fill else None,
              outline=A(outline, a) if outline else None, width=int(width * S))


def line(d, pts, col, a=1.0, width=4):
    if a > 0.01 and len(pts) > 1:
        d.line([tuple(P(*p)) for p in pts], fill=A(col, a), width=max(1, int(width * S)), joint="curve")


def rrect(d, x0, y0, x1, y1, r, fill=None, a=1.0, outline=None, width=0):
    if a > 0.01:
        d.rounded_rectangle(P(x0, y0, x1, y1), radius=int(r * S), fill=A(fill, a) if fill else None,
                            outline=A(outline, a) if outline else None, width=int(width * S))


def pill(d, s, x, y, size, col=WHITE, bg=NAVY, a=1.0, scale=1.0):
    if a <= 0.01 or scale <= 0.02:
        return
    size = size * scale
    w = F("bold", max(int(size), 2)).getlength(s) / S
    rrect(d, x - w / 2 - size * 0.6, y - size * 0.85, x + w / 2 + size * 0.6, y + size * 0.85,
          size * 0.85, bg, a * 0.9)
    text(d, s, x, y, int(size), col, a)


def gradient(top, bottom, mid=None):
    """竖直渐变背景（先生成单列，再横向拉伸）。"""
    y = np.linspace(0, 1, H * S)[:, None]
    if mid is None:
        col = np.array(top)[None, :] * (1 - y) + np.array(bottom)[None, :] * y
    else:
        u = np.clip(y * 2, 0, 1)
        v = np.clip(y * 2 - 1, 0, 1)
        col = np.where(y < 0.5, np.array(top) * (1 - u) + np.array(mid) * u,
                       np.array(mid) * (1 - v) + np.array(bottom) * v)
    strip = Image.fromarray(col.astype(np.uint8)[:, None, :].repeat(1, axis=1))
    return strip.resize((W * S, H * S), Image.NEAREST)


def cute_sun(d, x, y, r, t, col=SUNY, a=1.0, face=True):
    for k in range(12):
        ang = t * 0.4 + k * math.pi / 6
        r0, r1 = r * 1.22, r * (1.5 + 0.06 * math.sin(t * 3 + k))
        line(d, [(x + r0 * math.cos(ang), y + r0 * math.sin(ang)),
                 (x + r1 * math.cos(ang), y + r1 * math.sin(ang))], col, a, r * 0.13)
    circle(d, x, y, r * 1.08, col, a * 0.35)
    circle(d, x, y, r, col, a)
    if face:
        blink = 0.15 if (t % 3.7) < 0.12 else 1.0
        for sx in (-1, 1):
            d.ellipse(P(x + sx * r * 0.32 - r * 0.07, y - r * 0.12 - r * 0.11 * blink,
                        x + sx * r * 0.32 + r * 0.07, y - r * 0.12 + r * 0.11 * blink), fill=A(INK, a))
            circle(d, x + sx * r * 0.52, y + r * 0.2, r * 0.13, (255, 140, 120), a * 0.6)
        d.arc(P(x - r * 0.3, y - r * 0.05, x + r * 0.3, y + r * 0.4), 20, 160, fill=A(INK, a),
              width=int(r * 0.07 * S))


def cloud(d, x, y, s, a=1.0, col=WHITE):
    for dx, dy, r in [(-0.9, 0.15, 0.55), (-0.3, -0.25, 0.75), (0.45, -0.1, 0.65), (1.0, 0.2, 0.45)]:
        circle(d, x + dx * s, y + dy * s, r * s, col, a)
    rrect(d, x - 1.35 * s, y, x + 1.4 * s, y + 0.62 * s, 0.3 * s, col, a)


def kid(d, x, y, a=1.0, look_up=True, scale=1.0):
    """站立的小朋友，(x, y) 为脚底中点。"""
    s = scale
    rrect(d, x - 34 * s, y - 120 * s, x + 34 * s, y - 20 * s, 26 * s, (255, 120, 100), a)       # 身体
    line(d, [(x - 14 * s, y - 25 * s), (x - 16 * s, y)], INK, a, 12 * s)                      # 腿
    line(d, [(x + 14 * s, y - 25 * s), (x + 16 * s, y)], INK, a, 12 * s)
    hx, hy = x, y - 165 * s
    circle(d, hx, hy, 48 * s, (255, 222, 190), a)                                          # 头
    d.chord(P(hx - 50 * s, hy - 52 * s, hx + 50 * s, hy + 18 * s), 180, 360, fill=A((70, 50, 40), a))
    ey = hy - (6 if look_up else -6) * s
    for sx in (-1, 1):
        circle(d, hx + sx * 17 * s, ey + 8 * s, 6 * s, INK, a)
    d.arc(P(hx - 12 * s, hy + 12 * s, hx + 12 * s, hy + 30 * s), 20, 160, fill=A(INK, a), width=int(3 * s * S))


def wave_icon(d, x, y, length, wl, col, a, t, amp=22):
    pts = [(x + u, y + amp * math.sin(2 * math.pi * u / wl - t * 4)) for u in range(0, int(length) + 1, 4)]
    line(d, pts, col, a, 7)


def molecule(d, x, y, r, col, a, t, seed, face=True):
    ang = 0.6 * math.sin(t * 1.3 + seed)
    dx, dy = r * 0.75 * math.cos(ang), r * 0.75 * math.sin(ang)
    circle(d, x - dx, y - dy, r, col, a, WHITE, 3)
    circle(d, x + dx, y + dy, r, col, a, WHITE, 3)
    if face:
        for sx in (-0.3, 0.3):
            circle(d, x + dx + sx * r, y + dy - 0.1 * r, r * 0.13, INK, a)



# ---------------------------------------------------------------- 系列通用元素
def bg_day():
    return gradient(SKY_TOP, SKY_LOW)


def next_episode(d, t, t0, title, y=1290):
    """片尾“下期预告”。"""
    a = vk.fade(t, t0, 0.5)
    pill(d, "下期预告", 540, y - 70, 30, NAVY, SUNY, a, vk.pop(t, t0))
    text(d, title, 540, y, 50, WHITE, a, "fun", stroke=7)


def subtitle(d, sc, t):
    lines = sc["lines"]
    for i, ln in enumerate(lines):
        t_end = lines[i + 1]["t0"] if i + 1 < len(lines) else sc["end"] - 0.1
        if ln["t0"] - 0.08 <= t < t_end:
            a = min(vk.fade(t, ln["t0"] - 0.08, 0.12), 1 - vk.fade(t, t_end - 0.1, 0.1))
            size = 54
            while True:                                              # 两行仍放不下时缩小字号
                rows = vk.wrap_balanced(ln["show"], lambda s: F("bold", size).getlength(s) / S, 900,
                                        "，、：；！？")
                if size <= 40 or max(F("bold", size).getlength(r) / S for r in rows) <= 980:
                    break
                size -= 4
            y0 = 1470 - (len(rows) - 1) * size * 0.7
            for k, r in enumerate(rows):
                text(d, r, 540, y0 + k * size * 1.4, size, WHITE, a, stroke=7)


# ---------------------------------------------------------------- 渲染框架
_bg_day = None


def render(i, subs=True):
    global GA
    t = i / FPS
    sc = next((s for s in TL if s["start"] <= t < s["end"]), TL[-1])
    lt = t - sc["start"]
    ts = [ln["t0"] - sc["start"] for ln in sc["lines"]]
    bgf = getattr(EP, "BG", {}).get(sc["key"])
    img = bgf(lt, ts) if bgf else _bg_day.copy()
    d = ImageDraw.Draw(img, "RGBA")
    first, last = sc is TL[0], sc is TL[-1]
    GA = min(1.0 if first else vk.fade(lt, 0, 0.35), 1.0 if last else 1 - vk.fade(t, sc["end"] - 0.35, 0.35))
    EP.DRAW[sc["key"]](d, lt, ts)
    GA = 1.0
    text(d, f"{SERIES} · 第 {EP.NUMBER} 期", 540, 175, 30, WHITE, 0.9, stroke=4)
    if subs:
        subtitle(d, sc, t)
    return img.reduce(S).tobytes()


def init_worker(args, tl):
    global ARGS, TL, _bg_day
    ARGS, TL = args, tl
    _bg_day = bg_day()
    if hasattr(EP, "setup"):
        EP.setup(tl)


def save_covers(ep, prefix):
    """封面：9:16 原图、3:4 裁切、4:3（虚化背景上居中放置 3:4 图）。均不含字幕。"""
    from PIL import ImageFilter
    prefix = os.path.splitext(prefix)[0]
    img = Image.frombytes("RGB", (W, H), render(int(getattr(ep, "COVER_T", 1.2) * FPS), subs=False))
    img.save(prefix + ".jpg", quality=92)
    top = getattr(ep, "COVER_TOP", 110)
    c34 = img.crop((0, top, 1080, top + 1440))
    c34.save(prefix + "_3x4.jpg", quality=92)
    bg = c34.resize((1440, 1920)).crop((0, 420, 1440, 1500)).filter(ImageFilter.GaussianBlur(30))
    bg.paste(c34.resize((810, 1080)), (315, 0))
    bg.save(prefix + "_4x3.jpg", quality=92)


def run(ep, here):
    """解析命令行，合成语音，渲染并编码。here 为该期目录。"""
    global ARGS, EP
    EP = ep
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(ROOT, "assets"))
    ap.add_argument("--out", default=os.path.join(here, "video.mp4"))
    ap.add_argument("--cover", default=None, help="封面文件名前缀，如 cover → cover.jpg / cover_3x4.jpg / cover_4x3.jpg")
    ap.add_argument("--cover-only", action="store_true")
    ap.add_argument("--preview", type=float, nargs="*", default=None)
    ap.add_argument("--voice", default=os.environ.get("MINIMAX_VOICE", "Chinese (Mandarin)_Gentle_Senior"))
    ap.add_argument("--model", default=os.environ.get("MINIMAX_MODEL", "speech-2.6-hd"))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--emotion", default="happy")
    ARGS = ap.parse_args()

    tts = vk.make_tts(ARGS.assets, ARGS.voice, ARGS.model, ARGS.speed, ARGS.emotion)
    voice = vk.synth_lines(ep.SCENES, tts, os.path.join(ARGS.assets, "tts_cache"))
    tl, total = vk.build_timeline(ep.SCENES, voice, lead=0.45, gap=0.32, tail=0.6, final_hold=3.0)
    print(f"第 {ep.NUMBER} 期《{ep.TITLE}》 总时长 {total:.1f} s")
    for s in tl:
        print(f"  {s['key']:8s} {s['start']:6.2f} – {s['end']:6.2f}")
    light = vk.strip_audio(tl)

    if ARGS.cover_only:
        init_worker(ARGS, light)
        save_covers(ep, os.path.join(here, ARGS.cover or "cover"))
        return

    if ARGS.preview is not None:
        init_worker(ARGS, light)
        for p in ARGS.preview:
            Image.frombytes("RGB", (W, H), render(int(p * FPS))).save(
                os.path.splitext(ARGS.out)[0] + f"_{p:.1f}s.png")
        return

    wav = os.path.splitext(ARGS.out)[0] + ".wav"
    vk.write_mix(tl, total, lambda n: music(n, tl), wav)
    vk.encode(render, int(total * FPS), wav, ARGS.out, (W, H), FPS, init_worker, (ARGS, light))
    os.remove(wav)
    if ARGS.cover:
        init_worker(ARGS, light)
        save_covers(ep, os.path.join(here, ARGS.cover))
    print("完成：", ARGS.out)


# ---------------------------------------------------------------- 背景音乐
def music(n, tl):
    """轻快的五声音阶拨弦（约 104 BPM）+ 低音 + 场景切换处的提示音。"""
    SR = vk.SR
    t = np.arange(n) / SR
    out = np.zeros(n)
    beat = 60 / 104
    hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    chords = [[60, 64, 67], [57, 60, 64], [53, 57, 60], [55, 59, 62]]      # C Am F G
    penta = [0, 2, 4, 7, 9]
    rng = np.random.default_rng(1)
    nbeats = int(n / SR / beat)
    for b in range(nbeats):
        ch = chords[(b // 4) % 4]
        s = int(b * beat * SR)
        if b % 2 == 0:                                                     # 低音
            bt = np.arange(min(int(beat * 1.6 * SR), n - s)) / SR
            out[s:s + len(bt)] += 0.5 * np.sin(2 * np.pi * hz(ch[0] - 12) * bt) * np.exp(-bt / 0.35)
        for half in (0, 1):                                                # 八分音符拨弦
            if rng.random() < (0.85 if half == 0 else 0.45):
                m = 72 + penta[rng.integers(0, 5)] + (12 if rng.random() < 0.15 else 0)
                p = vk.pluck(hz(m), 0.9, 0.993)
                s2 = s + int(half * beat / 2 * SR)
                e = min(s2 + len(p), n)
                out[s2:e] += 0.28 * p[:e - s2]
    for sc in tl[1:]:                                                     # 场景切换的“叮”
        s = int(sc["start"] * SR)
        bt = np.arange(min(int(0.8 * SR), n - s)) / SR
        out[s:s + len(bt)] += 0.35 * (np.sin(2 * np.pi * 1568 * bt) + 0.3 * np.sin(2 * np.pi * 3136 * bt)) * np.exp(-bt / 0.25)
    out /= np.max(np.abs(out)) + 1e-9
    total = n / SR
    return out * np.minimum(1, np.minimum(t / 0.8, (total - t) / 2.0)) * 0.11
