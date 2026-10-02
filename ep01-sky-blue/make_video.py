"""
少儿科普短视频《天为什么是蓝色的？》

画面中的天空色、晚霞色与太阳颜色均由 optics.py 按瑞利散射计算得到；
“光粒子撞分子”一段为蒙特卡罗模拟，蓝光与红光的散射概率之比取 (700/450)^4 ≈ 5.9。

用法：
  export MINIMAX_API_KEY=...          # 可选；不设置则使用离线语音
  python3 make_video.py [--assets DIR] [--out OUT.mp4] [--cover COVER.jpg]
  python3 make_video.py --preview 12.5
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import vidkit as vk  # noqa: E402
import optics  # noqa: E402

W, H, FPS, S = 1080, 1920, 30, 2
SERIES = "好奇心实验室 · 第 1 期"

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

# ---------------------------------------------------------------- 文案
SCENES = [
    dict(key="hook", lines=[
        dict(show="小朋友，抬头看看天空。"),
        dict(show="天，为什么是蓝色的呢？"),
    ]),
    dict(key="prism", lines=[
        dict(show="先说说阳光。阳光看起来是白色的，"),
        dict(show="其实里面藏着红橙黄绿青蓝紫，好多种颜色。"),
    ]),
    dict(key="air", lines=[
        dict(show="天空里有数不清的空气分子，"),
        dict(show="它们小极了，比头发丝还要细十几万倍。"),
    ]),
    dict(key="scatter", lines=[
        dict(show="阳光照进空气，会撞上这些小分子。"),
        dict(show="红光的波浪长长的，大多直直地穿了过去；"),
        dict(show="蓝光的波浪短短的，特别容易被弹开，"),
        dict(show="被弹开的次数，差不多是红光的六倍！", pause=0.6),
    ]),
    dict(key="eye", lines=[
        dict(show="这些被弹开的蓝光，从四面八方跑进我们的眼睛，"),
        dict(show="所以不管往哪儿看，天都是蓝蓝的。"),
    ]),
    dict(key="violet", lines=[
        dict(show="咦？紫光的波浪更短，为什么天不是紫色的？"),
        dict(show="因为阳光里的紫光本来就少，我们的眼睛对紫色也不太敏感。"),
    ]),
    dict(key="sunset", lines=[
        dict(show="到了傍晚，太阳斜斜地照过来，"),
        dict(show="阳光要穿过厚厚的空气，蓝光在半路就被弹光了，"),
        dict(show="剩下的橙光和红光，就把天空染成了晚霞。"),
    ]),
    dict(key="end", lines=[
        dict(show="蓝天和晚霞，原来是同一个道理。"),
        dict(show="它有个名字，叫瑞利散射。"),
        dict(show="考考你：火星上的晚霞，是什么颜色的？"),
    ]),
]

ARGS = None
GA = 1.0
TL = None
SIM = None


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


# ---------------------------------------------------------------- 光子散射模拟
BOX = (80, 470, 1000, 1110)


def simulate(duration, seed=4, rate_blue=0.95, speed=330.0, emit=20.0):
    """二维蒙特卡罗：光子自左向右飞行，每帧以 rate·dt 的概率被散射到随机方向。"""
    rng = np.random.default_rng(seed)
    nf = int(duration * FPS) + 2
    dt = 1.0 / FPS
    rates = {"b": rate_blue, "r": rate_blue / optics.SCATTER_RATIO}
    photons, events, alive = [], [], []
    next_emit, k = 0.25, 0
    for f in range(nf):
        t = f * dt
        while t >= next_emit:
            c = "b" if k % 2 == 0 else "r"
            p = dict(c=c, birth=f, pos=[], x=BOX[0], y=rng.uniform(640, 940), vx=speed, vy=0.0)
            photons.append(p)
            alive.append(p)
            next_emit += 1.0 / emit
            k += 1
        for p in list(alive):
            if rng.random() < rates[p["c"]] * dt:
                th = rng.uniform(0, 2 * math.pi)
                p["vx"], p["vy"] = speed * math.cos(th), speed * math.sin(th)
                events.append((f, p["c"], p["x"], p["y"]))
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["pos"].append((p["x"], p["y"]))
            if not (BOX[0] - 5 <= p["x"] <= BOX[2] + 5 and BOX[1] - 5 <= p["y"] <= BOX[3] + 5):
                alive.remove(p)
    for p in photons:
        p["pos"] = np.array(p["pos"])
    return photons, events


# ---------------------------------------------------------------- 各场景
def bg_day():
    return gradient(SKY_TOP, SKY_LOW)


def sc_hook(d, t, ts):
    cloud(d, 180 + 18 * t, 560, 70, 0.95)
    cloud(d, 760 - 12 * t, 700, 55, 0.9)
    cute_sun(d, 830, 420, 95, t)
    d.ellipse(P(-400, 1270, 1480, 2600), fill=A(GRASS))
    kid(d, 300, 1330, 1.0)
    sc = vk.pop(t, 0.05, 0.6)
    text(d, "天为什么", 540, 860, 150 * sc, WHITE, 1.0, "fun", stroke=10)
    text(d, "是蓝色的？", 540, 1040, 150 * sc, WHITE, 1.0, "fun", stroke=10)


def sc_prism(d, t, ts):
    rrect(d, 60, 330, 1020, 1210, 44, PANEL, 1.0)
    text(d, "把阳光拆开看看", 540, 410, 56, WHITE, vk.fade(t, 0.2), "fun")
    # 白光入射
    p = vk.ease_out((t - ts[0]) / 0.8)
    x0, y0, x1, y1 = 90, 830, 395, 760
    if p > 0:
        xe, ye = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
        line(d, [(x0, y0), (xe, ye)], WHITE, 0.25, 34)
        line(d, [(x0, y0), (xe, ye)], WHITE, 0.95, 12)
    # 三棱镜
    cx, cy, sd = 470, 800, 300
    tri = [(cx, cy - sd * 0.577), (cx - sd / 2, cy + sd * 0.289), (cx + sd / 2, cy + sd * 0.289)]
    d.polygon([tuple(P(*q)) for q in tri], fill=A((210, 230, 255), 0.25), outline=A(WHITE, 0.9),
              width=int(4 * S))
    # 色散
    q = vk.ease_out((t - ts[1] + 0.1) / 1.0)
    ex, ey = 545, 760
    sx = 860
    if q > 0:
        for lam in range(700, 400, -5):
            ya, yb = 560 + (700 - lam) / 300 * 470, 560 + (700 - lam + 5) / 300 * 470
            col = optics.wavelength_rgb(lam - 2.5)
            pa = (ex + (sx - ex) * q, ey + (ya - ey) * q)
            pb = (ex + (sx - ex) * q, ey + (yb - ey) * q)
            d.polygon([tuple(P(ex, ey)), tuple(P(*pa)), tuple(P(*pb))], fill=A(col, 0.9))
        line(d, [(sx + 6, 555), (sx + 6, 1035)], WHITE, q * 0.6, 4)
    for i, (name, lam) in enumerate([("红", 680), ("橙", 615), ("黄", 580), ("绿", 530),
                                     ("青", 495), ("蓝", 460), ("紫", 415)]):
        y = 560 + (700 - lam) / 300 * 470
        sc = vk.pop(t, ts[1] + 1.0 + 0.28 * i)
        circle(d, 930, y, 26 * sc, optics.wavelength_rgb(lam), 1.0)
        text(d, name, 930, y, 30 * sc, INK if lam > 470 else WHITE, 1.0)


def sc_air(d, t, ts):
    cx, cy, r = 540, 780, 320
    a = vk.fade(t, 0, 0.4)
    line(d, [(cx + 0.72 * r, cy + 0.72 * r), (cx + 1.18 * r, cy + 1.18 * r)], (120, 80, 60), a, 46)
    circle(d, cx, cy, r, (232, 242, 255), a, NAVY, 20)
    rng = np.random.default_rng(5)
    for i in range(13):
        ang, rr = rng.uniform(0, 2 * math.pi), r * 0.78 * math.sqrt(rng.uniform(0.05, 1))
        x = cx + rr * math.cos(ang) + 12 * math.sin(t * 1.7 + i)
        y = cy + rr * math.sin(ang) + 12 * math.cos(t * 1.4 + i * 2)
        col = (90, 140, 230) if i % 5 else (240, 110, 100)
        molecule(d, x, y, 24, col, vk.fade(t, 0.2 + 0.08 * i, 0.4), t, i)
    pill(d, "氮气分子", 300, 470, 34, WHITE, (70, 110, 200), vk.fade(t, ts[0] + 1.0), vk.pop(t, ts[0] + 1.0))
    pill(d, "氧气分子", 790, 470, 34, WHITE, (220, 95, 90), vk.fade(t, ts[0] + 1.3), vk.pop(t, ts[0] + 1.3))
    pill(d, "比头发丝细十几万倍", 540, 1230, 42, NAVY, WHITE, vk.fade(t, ts[1] + 0.3), vk.pop(t, ts[1] + 0.3))


def sc_scatter(d, t, ts):
    rrect(d, BOX[0] - 20, BOX[1] - 20, BOX[2] + 20, BOX[3] + 20, 36, WHITE, 0.18, WHITE, 3)
    rng = np.random.default_rng(9)
    for i in range(70):
        x, y = rng.uniform(BOX[0] + 20, BOX[2] - 20), rng.uniform(BOX[1] + 20, BOX[3] - 20)
        circle(d, x + 4 * math.sin(t * 2 + i), y + 4 * math.cos(t * 1.7 + i), 7, WHITE, 0.55)
    photons, events = SIM
    f = int(t * FPS)
    for p in photons:
        k = f - p["birth"]
        if k < 0 or k >= len(p["pos"]):
            continue
        col = BLUE if p["c"] == "b" else RED
        tr = p["pos"][max(0, k - 9):k + 1]
        if len(tr) > 1:
            line(d, [tuple(q) for q in tr], col, 0.45, 6)
        x, y = p["pos"][k]
        circle(d, x, y, 17, col, 0.3)
        circle(d, x, y, 10, col, 1.0)
    for (ef, c, x, y) in events:          # 被弹开时的小圆圈
        age = (f - ef) / FPS
        if 0 <= age < 0.35:
            circle(d, x, y, 14 + 70 * age, None, 1 - age / 0.35, BLUE if c == "b" else RED, 3)
    # 波长示意
    a1 = vk.fade(t, ts[1])
    wave_icon(d, 110, 400, 330, 160, RED, a1, t)
    text(d, "红光：波浪长", 275, 345, 34, WHITE, a1, stroke=5)
    a2 = vk.fade(t, ts[2])
    wave_icon(d, 620, 400, 330, 55, BLUE, a2, t)
    text(d, "蓝光：波浪短", 785, 345, 34, WHITE, a2, stroke=5)
    # 计数
    a3 = vk.fade(t, ts[3] - 0.2)
    if a3 > 0:
        nb = sum(1 for e in events if e[0] <= f and e[1] == "b")
        nr = sum(1 for e in events if e[0] <= f and e[1] == "r")
        rrect(d, 120, 1150, 960, 1250, 30, NAVY, a3 * 0.85)
        text(d, f"被弹开   蓝光 {nb} 次   红光 {nr} 次", 540, 1200, 40, WHITE, a3)
        text(d, "瑞利散射：散射强度 ∝ 1/波长⁴　450 nm 蓝光约为 700 nm 红光的 5.9 倍",
             540, 1292, 26, WHITE, a3, "reg", stroke=3)


def sc_eye(d, t, ts):
    cx, cy = 540, 960
    rng = np.random.default_rng(2)
    for i in range(46):
        th = rng.uniform(math.radians(195), math.radians(345))
        sp = rng.uniform(0.35, 0.6)
        ph = (t * sp + rng.uniform(0, 1)) % 1.0
        r = 600 * (1 - ph) + 120
        a = vk.fade(t, 0.2) * min(1, ph * 4) * min(1, (1 - ph) * 5)
        x, y = cx + r * math.cos(th), cy + r * math.sin(th)
        x2, y2 = cx + (r + 45) * math.cos(th), cy + (r + 45) * math.sin(th)
        col = vk.lerp_col((40, 110, 255), (120, 190, 255), rng.uniform(0, 1))
        line(d, [(x, y), (x2, y2)], col, a, 9)
        circle(d, x, y, 7, col, a)
    # 眼睛
    blink = 0.12 if (t % 3.3) < 0.12 else 1.0
    ew, eh = 230, 130 * blink
    pts = [(cx + ew * math.cos(u), cy + eh * math.sin(u) * (1.0 if math.sin(u) < 0 else 0.85))
           for u in np.linspace(0, 2 * math.pi, 80)]
    d.polygon([tuple(P(*q)) for q in pts], fill=A(WHITE), outline=A(NAVY), width=int(8 * S))
    if blink > 0.5:
        circle(d, cx, cy - 10, 78, (70, 140, 230))
        circle(d, cx, cy - 10, 36, INK)
        circle(d, cx + 22, cy - 34, 14, WHITE)
    # 计算得到的天空色
    a = vk.fade(t, ts[1])
    rrect(d, 150, 1180, 930, 1290, 55, WHITE, a * 0.9)
    circle(d, 215, 1235, 40, SKY, a, NAVY, 3)
    text(d, "按散射规律计算出的天空颜色", 580, 1235, 38, NAVY, a)


def sc_violet(d, t, ts):
    rrect(d, 60, 330, 1020, 1240, 44, WHITE, 0.92)
    text(d, "为什么不是紫色？", 540, 420, 58, VIOLET, vk.fade(t, 0.1), "fun")
    x0, x1, y0, y1 = 150, 930, 560, 1010
    lam = np.arange(380, 721, 4.0)
    X = lambda l: x0 + (l - 380) / 340 * (x1 - x0)
    for l in lam:                                     # 色带
        d.rectangle(P(X(l), 1030, X(l + 4) + 0.5, 1075), fill=A(optics.wavelength_rgb(l + 2), vk.fade(t, 0.3)))
    text(d, "紫", X(390), 1110, 32, VIOLET, vk.fade(t, 0.3))
    text(d, "红", X(710), 1110, 32, RED, vk.fade(t, 0.3))
    band = vk.fade(t, ts[1] + 0.2)
    d.rectangle(P(X(380), y0 - 20, X(435), 1075), fill=A((150, 90, 230), 0.14 * band))
    sunv = optics.sun(lam)
    sunv = sunv / sunv.max()
    eye = optics.cmf(lam)[1]
    eye = eye / eye.max()
    for k, (vals, col, label, lx, ly) in enumerate([
            (sunv, (245, 150, 30), "阳光里的多少", X(640), y0 - 30),
            (eye, (40, 170, 90), "眼睛的敏感程度", X(445), y0 + 330)]):
        p = vk.ease((t - ts[1] - 0.2 - 1.6 * k) / 1.4)
        n = max(2, int(len(lam) * p))
        pts = [(X(l), y1 - v * (y1 - y0)) for l, v in zip(lam[:n], vals[:n])]
        line(d, pts, col, 1.0 if p > 0 else 0, 8)
        text(d, label, lx, ly, 34, col, vk.fade(t, ts[1] + 0.6 + 1.6 * k))
    line(d, [(x0, y1), (x1, y1)], (180, 185, 200), vk.fade(t, 0.3), 3)
    pill(d, "紫光：少，又不敏感", X(470), 1180, 34, WHITE, (140, 80, 220),
         vk.fade(t, ts[1] + 3.6), vk.pop(t, ts[1] + 3.6))


EARTH = (540, 2350, 1250)
ATMOS = 1440


def sunset_state(t, ts):
    span = (ts[2] + 0.6) - ts[0]
    z = 32 + 58 * vk.ease((t - ts[0]) / span)
    return z, float(optics.airmass(min(z, 90.0)))


def bg_sunset(t, ts):
    z, m = sunset_state(t, ts)
    s = vk.ease((z - 70) / 20)
    top = vk.lerp_col(SKY_TOP, DUSK_TOP, s)
    low = vk.lerp_col(SKY_LOW, optics.sky_color(m), s)
    mid = vk.lerp_col(top, low, 0.45 + 0.2 * s)
    return gradient(top, low, mid)


def sc_sunset(d, t, ts):
    z, m = sunset_state(t, ts)
    ex, ey, er = EARTH
    circle(d, ex, ey, ATMOS, WHITE, 0.16)
    ox, oy = 380, ey - math.sqrt(er ** 2 - (380 - ex) ** 2)
    R = 560
    sx, sy = ox + R * math.sin(math.radians(z)), oy - R * math.cos(math.radians(z))
    scol = optics.sun_color(m)
    # 阳光在大气中的路径
    dx, dy = ox - sx, oy - sy
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    fx, fy = sx - ex, sy - ey                          # 求射线与大气外缘的交点
    b = fx * ux + fy * uy
    c = fx * fx + fy * fy - ATMOS ** 2
    disc = b * b - c
    s_in = (-b - math.sqrt(disc)) if disc > 0 and (-b - math.sqrt(disc)) > 0 else 0.0
    px, py = sx + ux * s_in, sy + uy * s_in
    line(d, [(sx, sy), (px, py)], scol, 0.5, 10)
    line(d, [(px, py), (ox, oy)], scol, 0.9, 16)
    # 被弹走的蓝光
    a_b = vk.fade(t, ts[1]) * (1 - vk.fade(t, ts[2] + 1.0, 1.0))
    if a_b > 0:
        rng = np.random.default_rng(3)
        for i in range(26):
            u = rng.uniform(0.05, 0.95)
            ph = (t * 0.8 + rng.uniform(0, 1)) % 1
            nx, ny = (-uy, ux) if i % 2 else (uy, -ux)
            bx = px + (ox - px) * u + nx * 160 * ph
            by = py + (oy - py) * u + ny * 160 * ph
            circle(d, bx, by, 8, BLUE, a_b * (1 - ph))
    d.ellipse(P(ex - er, ey - er, ex + er, ey + er), fill=A(GRASS))
    kid(d, ox, oy + 6, 1.0, scale=0.8)
    cute_sun(d, sx, sy, 72, t, scol)
    # 标注
    a = vk.fade(t, ts[0] + 0.5)
    rrect(d, 560, 330, 1000, 470, 30, NAVY, a * 0.8)
    text(d, "阳光穿过的空气", 780, 370, 32, WHITE, a)
    text(d, f"约为正午的 {m:.0f} 倍", 780, 425, 38, SUNY, a)
    ca = vk.fade(t, ts[2] + 0.3, 1.0)
    cloud(d, 200, 560, 60, 0.9 * ca, vk.lerp_col(WHITE, optics.sky_color(m), 0.6))
    cloud(d, 760, 640, 48, 0.9 * ca, vk.lerp_col(WHITE, optics.sky_color(m), 0.6))


def bg_end(t, ts):
    sunset_low = optics.sky_color(float(optics.airmass(89.5)))
    return gradient(SKY_TOP, sunset_low, vk.lerp_col(SKY_LOW, (255, 214, 160), 0.5))


def sc_end(d, t, ts):
    mars = vk.fade(t, ts[2], 0.6)
    cute_sun(d, 220, 470, 60, t, SUNY, 1 - mars, face=False)
    cute_sun(d, 860, 1080, 60, t, optics.sun_color(30.0), 1 - mars, face=False)
    up = 1 - 0.35 * vk.ease((t - ts[2]) / 0.6)
    sc = vk.pop(t, ts[1] + 0.3, 0.6)
    y = 780 * up + 120 * (1 - up)
    text(d, "瑞利散射", 540, y, 140 * sc, WHITE, 1.0, "fun", stroke=10)
    text(d, "Rayleigh scattering · 1871", 540, y + 115, 34 * sc, WHITE, 1.0, "reg", stroke=4)
    if mars > 0:
        mx, my, mr = 540, 900, 150 * vk.pop(t, ts[2])
        circle(d, mx, my, mr * 1.1, (255, 170, 120), 0.3)
        circle(d, mx, my, mr, (196, 84, 40), 1.0)
        for cx_, cy_, cr in [(-0.35, -0.3, 0.18), (0.3, 0.2, 0.22), (-0.1, 0.45, 0.12)]:
            circle(d, mx + cx_ * mr, my + cy_ * mr, cr * mr, (160, 62, 30), 1.0)
        text(d, "？", mx + mr * 1.15, my - mr * 0.9, 120 * vk.pop(t, ts[2] + 0.4), WHITE, 1.0, "fun", stroke=8)
        text(d, "火星上的晚霞是什么颜色？", 540, 1170, 60, WHITE, mars, "fun", stroke=8)
        pill(d, "评论区告诉我", 540, 1270, 36, NAVY, WHITE, vk.fade(t, ts[2] + 1.5), vk.pop(t, ts[2] + 1.5))


DRAW = {"hook": sc_hook, "prism": sc_prism, "air": sc_air, "scatter": sc_scatter, "eye": sc_eye,
        "violet": sc_violet, "sunset": sc_sunset, "end": sc_end}
BG = {"sunset": bg_sunset, "end": bg_end}


# ---------------------------------------------------------------- 逐帧渲染
_bg_day = None


def subtitle(d, sc, t):
    lines = sc["lines"]
    for i, ln in enumerate(lines):
        t_end = lines[i + 1]["t0"] if i + 1 < len(lines) else sc["end"] - 0.1
        if ln["t0"] - 0.08 <= t < t_end:
            a = min(vk.fade(t, ln["t0"] - 0.08, 0.12), 1 - vk.fade(t, t_end - 0.1, 0.1))
            rows = vk.wrap_balanced(ln["show"], lambda s: F("bold", 54).getlength(s) / S, 900,
                                    "，、：；！？")
            y0 = 1470 - (len(rows) - 1) * 38
            for k, r in enumerate(rows):
                text(d, r, 540, y0 + k * 76, 54, WHITE, a, stroke=7)


def render(i):
    global GA
    t = i / FPS
    sc = next((s for s in TL if s["start"] <= t < s["end"]), TL[-1])
    lt = t - sc["start"]
    ts = [ln["t0"] - sc["start"] for ln in sc["lines"]]
    bgf = BG.get(sc["key"])
    img = bgf(lt, ts) if bgf else _bg_day.copy()
    d = ImageDraw.Draw(img, "RGBA")
    first, last = sc is TL[0], sc is TL[-1]
    GA = min(1.0 if first else vk.fade(lt, 0, 0.35), 1.0 if last else 1 - vk.fade(t, sc["end"] - 0.35, 0.35))
    DRAW[sc["key"]](d, lt, ts)
    GA = 1.0
    text(d, SERIES, 540, 175, 30, WHITE, 0.9, stroke=4)
    subtitle(d, sc, t)
    return img.reduce(S).tobytes()


def init_worker(args, tl):
    global ARGS, TL, SIM, _bg_day
    ARGS, TL = args, tl
    _bg_day = bg_day()
    sc = next(s for s in tl if s["key"] == "scatter")
    SIM = simulate(sc["end"] - sc["start"])


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(HERE, "assets"))
    ap.add_argument("--out", default=os.path.join(HERE, "sky_blue.mp4"))
    ap.add_argument("--cover", default=None)
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--voice", default=os.environ.get("MINIMAX_VOICE", "Chinese (Mandarin)_Gentle_Senior"))
    ap.add_argument("--model", default=os.environ.get("MINIMAX_MODEL", "speech-2.6-hd"))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--emotion", default="happy")
    ARGS = ap.parse_args()

    tts = vk.make_tts(ARGS.assets, ARGS.voice, ARGS.model, ARGS.speed, ARGS.emotion)
    voice = vk.synth_lines(SCENES, tts, os.path.join(ARGS.assets, "tts_cache"))
    tl, total = vk.build_timeline(SCENES, voice, lead=0.45, gap=0.32, tail=0.6, final_hold=3.0)
    print(f"总时长 {total:.1f} s")
    for s in tl:
        print(f"  {s['key']:8s} {s['start']:6.2f} – {s['end']:6.2f}")
    light = vk.strip_audio(tl)

    if ARGS.preview is not None:
        init_worker(ARGS, light)
        Image.frombytes("RGB", (W, H), render(int(ARGS.preview * FPS))).save(
            os.path.splitext(ARGS.out)[0] + f"_{ARGS.preview:.1f}s.png")
        return

    wav = os.path.splitext(ARGS.out)[0] + ".wav"
    vk.write_mix(tl, total, lambda n: music(n, tl), wav)
    vk.encode(render, int(total * FPS), wav, ARGS.out, (W, H), FPS, init_worker, (ARGS, light))
    os.remove(wav)
    if ARGS.cover:
        init_worker(ARGS, light)
        Image.frombytes("RGB", (W, H), render(int(1.2 * FPS))).convert("RGB").save(ARGS.cover, quality=92)
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


if __name__ == "__main__":
    main()
