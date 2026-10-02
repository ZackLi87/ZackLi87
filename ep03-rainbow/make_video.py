"""
第 3 期《彩虹为什么是弯的？》

水滴中的光线按斯涅尔定律逐条追迹（折射—内反射—折射），水的色散采用 Daimon & Masumura 公式
（rainbow.py）。彩虹图像由出射角分布计算：对每个波长按入射高度均匀取样（面积权重 b），
统计出射角直方图，以 0.25° 高斯平滑代表太阳视直径，再按太阳光谱与 CIE 色匹配函数合成颜色。
"""
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import kidkit as kk  # noqa: E402
import optics  # noqa: E402
import vidkit as vk  # noqa: E402
from rainbow import exit_angle, n_water, rainbow_angle  # noqa: E402

NUMBER = 3
TITLE = "彩虹为什么是弯的？"
COVER_T = 1.5

ANG_RED, ANG_VIO = rainbow_angle(700), rainbow_angle(400)

# ---------------------------------------------------------------- 彩虹颜色的计算
LAM = np.arange(400, 701, 5.0)
DTH = 0.05
TH_BINS = np.arange(0, 60 + DTH, DTH)


def bow_table():
    b = np.linspace(0, 1, 400001)
    g = np.exp(-0.5 * (np.arange(-15, 16) * DTH / 0.25) ** 2)
    I = np.zeros((len(LAM), len(TH_BINS) - 1))
    for i, l in enumerate(LAM):
        h, _ = np.histogram(exit_angle(b, l), bins=TH_BINS, weights=b)
        I[i] = np.convolve(h, g / g.sum(), "same")
    x, y, z = optics.cmf(LAM)
    spec = I * optics.sun(LAM)[:, None]
    XYZ = np.stack([(spec * c[:, None]).sum(0) for c in (x, y, z)])
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(M @ XYZ, 0, None).T
    rgb /= np.percentile(rgb.max(1), 99.5)
    return rgb                                                   # 线性 sRGB，按角度排列


BOW = bow_table()


def bow_layer(w, h, cx, cy, px_per_deg, gain=1.6):
    """返回（颜色图, 透明度蒙版）：像素颜色由其与反日点的夹角决定。"""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = np.hypot(xs - cx, ys - cy) / px_per_deg
    idx = np.clip((ang / DTH).astype(int), 0, len(BOW) - 1)
    lin = BOW[idx]
    lum = lin.max(-1)
    chroma = lin / np.maximum(lum, 1e-6)[..., None]
    srgb = np.where(chroma <= 0.0031308, 12.92 * chroma, 1.055 * np.power(chroma, 1 / 2.4) - 0.055)
    alpha = np.clip(lum * gain, 0, 0.9)
    return (Image.fromarray((srgb.clip(0, 1) * 255).astype(np.uint8)),
            Image.fromarray((alpha * 255).astype(np.uint8)))


# ---------------------------------------------------------------- 水滴中的光线追迹
def refract(I, N, eta):
    cosi = -np.dot(N, I)
    k = 1 - eta ** 2 * (1 - cosi ** 2)
    return eta * I + (eta * cosi - math.sqrt(max(k, 0))) * N


def trace(b, lam, cx, cy, R):
    """返回光路折点 [入射起点, 入射点, 反射点, 出射点] 与出射方向（屏幕坐标，y 向下）。"""
    n = float(n_water(lam))
    c = np.array([cx, cy], dtype=float)
    p1 = np.array([cx - R * math.sqrt(1 - b * b), cy - b * R])
    d0 = np.array([1.0, 0.0])
    d1 = refract(d0, (p1 - c) / R, 1 / n)
    p2 = p1 + (-2 * np.dot(p1 - c, d1)) * d1
    n2 = (p2 - c) / R
    d2 = d1 - 2 * np.dot(d1, n2) * n2
    p3 = p2 + (-2 * np.dot(p2 - c, d2)) * d2
    d3 = refract(d2, -(p3 - c) / R, n)
    return [np.array([60.0, cy - b * R]), p1, p2, p3], d3


def draw_path(d, pts, d3, col, a, prog, width=6, out_len=420):
    """按进度 prog（0–1）依次画出各段光路。"""
    segs = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[3] + d3 * out_len)]
    lens = [np.linalg.norm(q - p) for p, q in segs]
    remain = prog * sum(lens)
    for (p, q), L in zip(segs, lens):
        if remain <= 0:
            break
        u = min(1.0, remain / L)
        kk.line(d, [tuple(p), tuple(p + (q - p) * u)], col, a, width)
        remain -= L


# ---------------------------------------------------------------- 文案
SCENES = [
    dict(key="hook", lines=[
        dict(show="雨后天晴，天上挂出一道彩虹。"),
        dict(show="彩虹为什么总是弯弯的呢？"),
    ]),
    dict(key="drop", lines=[
        dict(show="彩虹的秘密，藏在一颗颗小水珠里。"),
        dict(show="阳光钻进水珠，先拐个弯，在水珠背面反射一次，再拐个弯钻出来。"),
    ]),
    dict(key="colors", lines=[
        dict(show="不同颜色的光，拐弯的角度不一样："),
        dict(show="红光从大约四十二度的方向出来，紫光大约是四十度。"),
    ]),
    dict(key="bunch", lines=[
        dict(show="我们让很多条光线射进水珠，算一算它们从哪里出来。"),
        dict(show="你看，光线在四十二度附近挤成一团，那里就特别亮。"),
    ]),
    dict(key="circle", lines=[
        dict(show="所以，只有和你的视线成四十二度角的水珠，才会把彩色的光送进你的眼睛。"),
        dict(show="这些水珠连起来，正好是一个圆。"),
    ]),
    dict(key="ground", lines=[
        dict(show="地面挡住了圆的下半部分，所以我们看到的彩虹是弯的。"),
        dict(show="在飞机上，有时还能看到一整圈圆形的彩虹呢！"),
    ]),
    dict(key="end", lines=[
        dict(show="记住：彩虹总是出现在背对太阳的方向。"),
        dict(show="下一期，我们来看看：肥皂泡为什么是彩色的？"),
    ]),
]

# ---------------------------------------------------------------- 画面
RAIN_TOP, RAIN_LOW = (92, 104, 128), (196, 206, 222)
PXD = 17.0                                    # 天空场景：每度对应的像素
_layers = {}


def setup(tl):
    S = kk.S
    _layers["hook"] = bow_layer(kk.W * S, kk.H * S, 540 * S, 1520 * S, PXD * S)
    _layers["circle"] = bow_layer(kk.W * S, kk.H * S, 540 * S, 1020 * S, 9.0 * S)
    _layers["plane"] = bow_layer(kk.W * S, kk.H * S, 540 * S, 820 * S, 8.0 * S)


def paste_bow(d, key, a):
    if a <= 0.01:
        return
    col, mask = _layers[key]
    m = mask if a >= 0.99 else mask.point(lambda v: int(v * a))
    d._image.paste(col, (0, 0), m)


def bg_rain(t, ts):
    return kk.gradient(RAIN_TOP, RAIN_LOW)


def drop(d, cx, cy, R, a=1.0):
    kk.circle(d, cx, cy, R, (190, 225, 255), a * 0.35, kk.WHITE, 6)
    kk.circle(d, cx - R * 0.45, cy - R * 0.5, R * 0.12, kk.WHITE, a * 0.7)


def sc_hook(d, t, ts):
    paste_bow(d, "hook", 0.55 + 0.45 * vk.fade(t, 0.0, 1.5))
    for i in range(40):                                               # 雨丝
        x = (i * 97 + t * 380) % 1100 - 10
        y = (i * 211 + t * 900) % 700 + 260
        kk.line(d, [(x, y), (x - 8, y + 40)], kk.WHITE, 0.35 * (1 - vk.fade(t, ts[0] + 1.0, 1.5)), 3)
    d.ellipse(kk.P(-400, 1250, 1480, 2600), fill=kk.A(kk.GRASS))
    kk.kid(d, 540, 1310, 1.0)
    sc = 1.0
    kk.text(d, "彩虹为什么", 540, 480, 130 * sc, kk.WHITE, 1.0, "fun", stroke=10)
    kk.text(d, "是弯的？", 540, 640, 130 * sc, kk.WHITE, 1.0, "fun", stroke=10)


DROP = (600, 660, 280)


def sc_drop(d, t, ts):
    cx, cy, R = DROP
    drop(d, cx, cy, R, vk.fade(t, 0.1))
    kk.pill(d, "放大的小水珠", cx, cy - R - 60, 34, kk.NAVY, kk.WHITE, vk.fade(t, 0.3), vk.pop(t, 0.3))
    pts, d3 = trace(0.86, 600, cx, cy, R)
    p = vk.ease((t - ts[1] - 0.2) / 3.0)
    draw_path(d, pts, d3, kk.WHITE, 1.0, p, 8)
    labels = [(0.25, pts[1], "① 拐弯", -150, -50), (0.55, pts[2], "② 反射", 40, -60), (0.8, pts[3], "③ 拐弯", 40, 40)]
    for thr, q, s, dx, dy in labels:
        kk.pill(d, s, q[0] + dx, q[1] + dy, 30, kk.WHITE, kk.NAVY, vk.fade(t, ts[1] + 0.2 + 3.0 * thr),
                vk.pop(t, ts[1] + 0.2 + 3.0 * thr))
    kk.cute_sun(d, 110, 330, 55, t)


def angle_arc(d, p, d3, deg, col, a, r=150):
    """在出射点画出“出射光与入射反方向的夹角”。"""
    base = math.degrees(math.atan2(0, -1))                           # 入射光的反方向（向左）
    ang_out = math.degrees(math.atan2(d3[1], d3[0]))
    kk.line(d, [tuple(p), (p[0] - r * 1.4, p[1])], (230, 235, 245), a, 3)
    lo, hi = sorted([base, ang_out])
    if hi - lo > 180:
        lo, hi = hi, lo + 360
    d.arc(kk.P(p[0] - r, p[1] - r, p[0] + r, p[1] + r), lo, hi, fill=kk.A(col, a), width=int(4 * kk.S))


def sc_colors(d, t, ts):
    cx, cy, R = DROP
    drop(d, cx, cy, R)
    for lam, col, ang, t0, dy in [(700, kk.RED, ANG_RED, ts[1], -50), (400, kk.VIOLET, ANG_VIO, ts[1] + 1.4, 50)]:
        pts, d3 = trace(0.86, lam, cx, cy, R)
        p = vk.ease((t - 0.2) / 1.8)
        draw_path(d, pts, d3, col, 1.0 if lam == 700 or t > ts[0] + 0.5 else 0, p, 7, 360)
        a = vk.fade(t, t0)
        end = pts[3] + d3 * 360
        dx = -150 if lam == 700 else 150
        kk.pill(d, f"{'红' if lam == 700 else '紫'}光 {ang:.1f}°", end[0] + dx, end[1] + dy, 34, kk.WHITE, col, a,
                vk.pop(t, t0))
    kk.text(d, "（按水的折射率计算）", 540, 300, 30, kk.WHITE, vk.fade(t, ts[1] + 2.5), stroke=4)


HB = np.linspace(0, 1, 20001)
HANG = exit_angle(HB, 600)
HEDGES = np.arange(0, 46.5, 0.75)
HMAX = np.histogram(HANG, bins=HEDGES, weights=HB)[0].max()


def sc_bunch(d, t, ts):
    cx, cy, R = 560, 640, 220
    drop(d, cx, cy, R)
    bs = np.linspace(0.02, 0.99, 48)
    n_show = int(len(bs) * vk.ease((t - 0.3) / max(ts[1] - 0.3, 0.5)))
    for b in bs[:n_show]:
        pts, d3 = trace(b, 600, cx, cy, R)
        draw_path(d, pts, d3, (255, 214, 120), 0.45, 1.0, 3, 360)
    # 出射角直方图
    x0, x1, y0, y1 = 120, 960, 1000, 1240
    kk.rrect(d, x0 - 30, y0 - 50, x1 + 30, y1 + 70, 26, kk.WHITE, 0.9)
    m = int(len(HB) * n_show / len(bs))
    h, edges = np.histogram(HANG[:m], bins=HEDGES, weights=HB[:m])
    for k, v in enumerate(h):
        if v > 0:
            xa = x0 + edges[k] / 46.5 * (x1 - x0)
            xb = x0 + edges[k + 1] / 46.5 * (x1 - x0) - 2
            d.rectangle(kk.P(xa, y1 - v / HMAX * (y1 - y0), xb, y1), fill=kk.A((245, 170, 40)))
    for v in (0, 10, 20, 30, 42):
        kk.text(d, f"{v}°", x0 + v / 46.5 * (x1 - x0), y1 + 30, 24, kk.INK, 1.0, "reg")
    kk.text(d, "光线射出的角度", 540, y0 - 22, 28, kk.INK)
    kk.pill(d, "42° 附近最亮", x0 + 30 / 46.5 * (x1 - x0), y0 + 30, 30, kk.WHITE, (230, 120, 30),
            vk.fade(t, ts[1] + 0.5), vk.pop(t, ts[1] + 0.5))


def sc_circle(d, t, ts):
    cx, cy, rp = 540, 1020, 42 * 9.0
    paste_bow(d, "circle", vk.fade(t, ts[1], 1.0))
    a = vk.fade(t, 0.2)
    kk.circle(d, cx, cy, 14, kk.INK, a)
    kk.text(d, "你的影子（正对着太阳的反方向）", cx, cy + 50, 28, kk.WHITE, a, stroke=4)
    sweep = vk.ease((t - ts[0] - 0.5) / 3.0) * 360
    for k in range(0, int(sweep), 15):
        th = math.radians(k - 90)
        x, y = cx + rp * math.cos(th), cy + rp * math.sin(th)
        kk.line(d, [(cx, cy), (x, y)], kk.WHITE, 0.25 * (1 - vk.fade(t, ts[1], 0.8)), 2)
        kk.circle(d, x, y, 12, (190, 225, 255), 1.0 - 0.6 * vk.fade(t, ts[1], 0.8), kk.WHITE, 2)
    kk.pill(d, "42°", cx + rp * 0.5 * math.cos(math.radians(-60)) + 30, cy + rp * 0.5 * math.sin(math.radians(-60)),
            32, kk.NAVY, kk.SUNY, a, vk.pop(t, 0.6))


def sc_ground(d, t, ts):
    plane = vk.fade(t, ts[1], 0.6)
    if plane < 1:
        paste_bow(d, "circle", 1 - plane)
        g = vk.ease((t - 0.3) / 1.5)
        d.rectangle(kk.P(0, 1920 - (1920 - 1030) * g, 1080, 1920), fill=kk.A(kk.GRASS, 1 - plane))
        kk.kid(d, 540, 1040, 1 - plane, scale=0.9)
    if plane > 0:
        d.rectangle(kk.P(0, 0, 1080, 1920), fill=kk.A((210, 222, 240), plane))
        for i in range(9):
            kk.cloud(d, 120 + (i % 3) * 420, 420 + (i // 3) * 420, 120, plane * 0.9)
        paste_bow(d, "plane", plane)
        sc = 0.9
        x, y = 540, 820
        d.polygon([tuple(kk.P(x + u * sc, y + v * sc)) for u, v in
                   [(0, -60), (12, -10), (80, 10), (80, 22), (12, 18), (8, 50), (24, 62), (-24, 62), (-8, 50),
                    (-12, 18), (-80, 22), (-80, 10), (-12, -10)]], fill=kk.A((70, 80, 100), plane * 0.8))
        kk.pill(d, "飞机的影子", x, y + 110, 30, kk.WHITE, kk.NAVY, plane)


def sc_end(d, t, ts):
    paste_bow(d, "hook", 1.0)
    d.ellipse(kk.P(-400, 1250, 1480, 2600), fill=kk.A(kk.GRASS))
    kk.kid(d, 540, 1310, 1.0)
    a = vk.fade(t, 0.3)
    kk.pill(d, "太阳在身后，彩虹在前方", 540, 420, 40, kk.NAVY, kk.WHITE, a, vk.pop(t, 0.3))
    kk.next_episode(d, t, ts[1] + 0.2, "肥皂泡为什么是彩色的？", y=640)


DRAW = {"hook": sc_hook, "drop": sc_drop, "colors": sc_colors, "bunch": sc_bunch, "circle": sc_circle,
        "ground": sc_ground, "end": sc_end}
BG = {"hook": bg_rain, "drop": bg_rain, "colors": bg_rain, "bunch": bg_rain, "circle": bg_rain,
      "ground": bg_rain, "end": bg_rain}

if __name__ == "__main__":
    print(f"虹角：红光 {ANG_RED:.2f}°，紫光 {ANG_VIO:.2f}°")
    kk.run(sys.modules[__name__], HERE)
