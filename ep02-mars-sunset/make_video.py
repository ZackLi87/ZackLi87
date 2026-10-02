"""
第 2 期《火星的晚霞为什么是蓝色的？》

火星日落的天空颜色由 mie.py 计算：火星尘埃取对数正态粒径分布（中值半径 1.5 μm），
复折射率实部 1.52、虚部随波长由 0.012（400 nm）降至 0.0015（700 nm），为文献量级的示意参数；
单次散射近似，颜色 = Σ 太阳光谱 × 散射强度(λ, θ)。
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
from mie import mie  # noqa: E402

NUMBER = 2
TITLE = "火星的晚霞为什么是蓝色的？"
COVER_T = 1.5

# ---------------------------------------------------------------- 米氏散射计算
LAM = np.arange(400, 701, 10.0)
TH = np.radians(np.concatenate([np.arange(0, 20, 0.1), np.arange(20, 181, 1.0)]))


def dust_scattering():
    k = np.interp(LAM, [400, 450, 550, 700], [0.012, 0.008, 0.004, 0.0015])
    radii = np.linspace(0.5, 3.0, 26)
    w = np.exp(-0.5 * (np.log(radii / 1.5) / 0.35) ** 2)
    I = np.zeros((len(LAM), len(TH)))
    for i, l in enumerate(LAM):
        for r, wi in zip(radii, w):
            S1, S2, _, _ = mie(1.52 + 1j * k[i], 2 * np.pi * r / (l / 1000), np.cos(TH))
            I[i] += wi * (abs(S1) ** 2 + abs(S2) ** 2) / 2 * (l / 1000) ** 2
    return I


I_DUST = dust_scattering()
SUN = optics.sun(LAM)


ANG = np.arange(0, 91, 0.25)
_spec = np.array([np.interp(ANG, np.degrees(TH), I_DUST[i]) for i in range(len(LAM))]) * SUN[:, None]
_k = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)                 # 约 0.75° 的高斯平滑，抹去粒径离散造成的条纹
_spec = np.array([np.convolve(np.pad(row, 8, mode="edge"), _k / _k.sum(), "valid") for row in _spec])
MARS_RGB = np.array([optics.to_srgb(_spec[:, j], LAM) for j in range(len(ANG))], dtype=float)
_lum = np.log10(_spec.T @ optics.cmf(LAM)[1])
MARS_LUM = (_lum - _lum.min()) / (_lum.max() - _lum.min())          # 0–1，对数亮度
MARS_DAY = tuple(int(v) for v in MARS_RGB[-1])
EARTH_DAY, EARTH_DUSK = kk.SKY, optics.sky_color(38.0)
BLUE_RATIO = I_DUST[np.argmin(abs(LAM - 450)), 0] / I_DUST[np.argmin(abs(LAM - 700)), 0]

RUST = (190, 92, 52)
DUST = (214, 128, 80)


def mars_sky_image(w, h, sx, sy, deg_per_px, dark=0.0):
    """按计算结果生成火星日落天空：每个像素的颜色由其与太阳的夹角决定。"""
    sm = 4
    ys, xs = np.mgrid[0:h // sm, 0:w // sm].astype(float) * sm
    ang = np.hypot(xs - sx, ys - sy) * deg_per_px
    idx = np.clip(ang / 0.25, 0, len(ANG) - 1).astype(int)
    rgb = MARS_RGB[idx] * (0.35 + 0.65 * MARS_LUM[idx])[..., None] * (1 - dark)
    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    return img.resize((w, h), Image.BICUBIC)


# ---------------------------------------------------------------- 文案
SCENES = [
    dict(key="hook", lines=[
        dict(show="上一期的问题：火星上的晚霞是什么颜色？"),
        dict(show="答案是：蓝色！火星车真的拍到过。"),
    ]),
    dict(key="compare", lines=[
        dict(show="地球的天空是蓝的，晚霞是红的；"),
        dict(show="火星正好反过来：白天天空是黄褐色，傍晚太阳周围却是蓝色。"),
    ]),
    dict(key="air", lines=[
        dict(show="火星的空气非常稀薄，还不到地球的百分之一，"),
        dict(show="却飘着大量细细的红色尘埃。"),
    ]),
    dict(key="size", lines=[
        dict(show="这些尘埃比空气分子大了几千倍，"),
        dict(show="比光的波浪还要长好几倍。"),
    ]),
    dict(key="lobe", lines=[
        dict(show="大颗粒散射光的方式不一样：它们喜欢把光往前推，"),
        dict(show="而且蓝光被推得更集中，都挤在太阳周围的一小圈里。"),
    ]),
    dict(key="sunset", lines=[
        dict(show="傍晚，阳光斜穿过长长的尘埃层，太阳周围就亮起一圈蓝光；"),
        dict(show="离太阳远一点的天空，被尘埃染成了黄褐色。"),
    ]),
    dict(key="end", lines=[
        dict(show="同样的阳光，遇到大小不同的颗粒，就会变出不同的颜色。"),
        dict(show="小分子的散射叫瑞利散射，大颗粒的散射叫米氏散射。"),
        dict(show="下一期，我们来看看：彩虹为什么是弯的？"),
    ]),
]


# ---------------------------------------------------------------- 画面
_sky_hook = None
_sky_sunset = None


def setup(tl):
    global _sky_hook, _sky_sunset
    _sky_hook = mars_sky_image(kk.W * kk.S, kk.H * kk.S, 540 * kk.S, 1180 * kk.S, 0.03 / kk.S)
    _sky_sunset = mars_sky_image(kk.W * kk.S, kk.H * kk.S, 620 * kk.S, 1060 * kk.S, 0.025 / kk.S)


def mars_ground(d, y0, a=1.0):
    pts = [(x, y0 + 30 * math.sin(x / 140) + 18 * math.sin(x / 47)) for x in range(0, 1081, 20)]
    d.polygon([tuple(kk.P(*p)) for p in pts + [(1080, 1920), (0, 1920)]], fill=kk.A(RUST, a))
    for x, y, r in [(150, y0 + 120, 22), (420, y0 + 200, 14), (820, y0 + 150, 30), (960, y0 + 260, 18)]:
        kk.circle(d, x, y, r, (150, 70, 40), a)


def rover(d, x, y, s=1.0, a=1.0):
    kk.rrect(d, x - 70 * s, y - 60 * s, x + 70 * s, y - 15 * s, 10 * s, (230, 230, 235), a)
    kk.line(d, [(x + 40 * s, y - 60 * s), (x + 40 * s, y - 120 * s)], (200, 200, 210), a, 8 * s)
    kk.rrect(d, x + 20 * s, y - 145 * s, x + 70 * s, y - 115 * s, 6 * s, (230, 230, 235), a)
    kk.circle(d, x + 55 * s, y - 130 * s, 8 * s, kk.INK, a)
    for wx in (-50, 0, 50):
        kk.circle(d, x + wx * s, y, 18 * s, kk.INK, a)


def blue_sun(d, x, y, r, t, a=1.0):
    for k in range(5, 0, -1):
        kk.circle(d, x, y, r * (1 + 0.9 * k), (150, 190, 255), a * 0.08)
    kk.circle(d, x, y, r, (240, 248, 255), a)


def sc_hook(d, t, ts):
    blue_sun(d, 540, 1180, 30, t)
    mars_ground(d, 1190)
    rover(d, 230, 1300, 1.1)
    sc = vk.pop(t, 0.05, 0.6)
    kk.text(d, "火星的晚霞", 540, 520, 130 * sc, kk.WHITE, 1.0, "fun", stroke=10)
    kk.text(d, "为什么是蓝色的？", 540, 680, 110 * sc, kk.WHITE, 1.0, "fun", stroke=10)
    kk.pill(d, "火星车拍到的日落", 540, 880, 38, kk.NAVY, kk.WHITE, vk.fade(t, ts[1] + 0.4),
            vk.pop(t, ts[1] + 0.4))


def panel_sky(d, x0, y0, x1, y1, top, low, a):
    """在矩形内画竖直渐变（逐行）。"""
    for k in range(0, int(y1 - y0), 4):
        col = vk.lerp_col(top, low, k / (y1 - y0))
        d.rectangle(kk.P(x0, y0 + k, x1, y0 + k + 4), fill=kk.A(col, a))


def sc_compare(d, t, ts):
    rows = [("地球", 380, EARTH_DAY, EARTH_DUSK, kk.SUNY, (255, 150, 60), ts[0]),
            ("火星", 840, MARS_DAY, MARS_DAY, (255, 240, 220), (235, 245, 255), ts[1])]
    for name, y, day, dusk, sun_day, sun_dusk, t0 in rows:
        a = vk.fade(t, t0)
        kk.text(d, name, 540, y - 30, 48, kk.WHITE, a, "fun", stroke=6)
        for j, (label, x0) in enumerate([("白天", 80), ("傍晚", 560)]):
            x1, yy0, yy1 = x0 + 440, y + 10, y + 330
            if name == "地球":
                top, low = (day, vk.lerp_col(day, kk.WHITE, 0.4)) if j == 0 else (kk.DUSK_TOP, dusk)
                panel_sky(d, x0, yy0, x1, yy1, top, low, a)
                kk.cute_sun(d, x0 + 330, yy0 + (90 if j == 0 else 260), 34, t, sun_day if j == 0 else sun_dusk,
                            a, face=False)
                d.rectangle(kk.P(x0, yy1 - 40, x1, yy1), fill=kk.A(kk.GRASS, a))
            else:
                if j == 0:
                    panel_sky(d, x0, yy0, x1, yy1, vk.lerp_col(MARS_DAY, RUST, 0.25), MARS_DAY, a)
                    kk.circle(d, x0 + 330, yy0 + 90, 26, sun_day, a)
                else:
                    tile = _sky_sunset.resize((440 * kk.S, 320 * kk.S), Image.BICUBIC, box=(
                        (620 - 440) * kk.S, (1060 - 280) * kk.S, (620 + 440) * kk.S, (1060 + 360) * kk.S))
                    if a > 0.99:
                        d._image.paste(tile, (int(x0 * kk.S), int(yy0 * kk.S)))
                    else:
                        panel_sky(d, x0, yy0, x1, yy1, MARS_DAY, MARS_DAY, a)
                    blue_sun(d, x0 + 220, yy1 - 60, 16, t, a)
                d.rectangle(kk.P(x0, yy1 - 40, x1, yy1), fill=kk.A(RUST, a))
            kk.rrect(d, x0, yy0, x1, yy1, 24, None, a, kk.WHITE, 5)
            kk.pill(d, label, x0 + 80, yy0 + 45, 28, kk.WHITE, kk.NAVY, a)


def sc_air(d, t, ts):
    for j, (name, x0, n_air, n_dust, col) in enumerate([("地球", 90, 160, 0, kk.SKY), ("火星", 570, 6, 18, MARS_DAY)]):
        a = vk.fade(t, 0.2 + 0.3 * j)
        kk.rrect(d, x0, 380, x0 + 420, 1150, 30, col, a * 0.85, kk.WHITE, 5)
        kk.text(d, name + "的空气", x0 + 210, 440, 44, kk.WHITE, a, "fun", stroke=6)
        rng = np.random.default_rng(j + 3)
        for i in range(n_air):
            x, y = rng.uniform(x0 + 25, x0 + 395), rng.uniform(500, 1130)
            kk.circle(d, x + 5 * math.sin(t * 2 + i), y + 5 * math.cos(t * 1.6 + i), 6, kk.WHITE, a * 0.9)
        ad = vk.fade(t, ts[1])
        for i in range(n_dust):
            x, y = rng.uniform(x0 + 40, x0 + 380), rng.uniform(520, 1110)
            r = rng.uniform(14, 24)
            d.regular_polygon((kk.P(x + 6 * math.sin(t + i), y)[0], kk.P(x, y + 6 * math.cos(t * 0.8 + i))[1],
                               int(r * kk.S)), 7, rotation=i * 20, fill=kk.A(DUST, ad), outline=kk.A(RUST, ad))
    kk.pill(d, "气压不到地球的 1%", 750, 1220, 36, kk.WHITE, kk.NAVY, vk.fade(t, ts[0] + 1.0),
            vk.pop(t, ts[0] + 1.0))


def sc_size(d, t, ts):
    kk.rrect(d, 70, 360, 1010, 1240, 40, kk.WHITE, 0.9)
    y = 760
    a0 = vk.fade(t, 0.2)
    kk.circle(d, 170, y, 5, kk.INK, a0)
    kk.text(d, "空气分子", 170, y + 70, 32, kk.INK, a0)
    kk.text(d, "约 0.0004 微米", 170, y + 115, 26, (110, 115, 130), a0, "reg")
    a1 = vk.fade(t, ts[1])
    kk.wave_icon(d, 290, y, 260, 90, kk.BLUE, a1, t)
    kk.text(d, "光的波浪", 420, y + 70, 32, kk.INK, a1)
    kk.text(d, "约 0.5 微米", 420, y + 115, 26, (110, 115, 130), a1, "reg")
    a2 = vk.fade(t, ts[0] + 0.3)
    s = vk.pop(t, ts[0] + 0.3, 0.6)
    if s > 0.02:
        d.regular_polygon((*kk.P(800, y), int(150 * s * kk.S)), 9, rotation=10, fill=kk.A(DUST, a2),
                          outline=kk.A(RUST, a2), width=int(5 * kk.S))
    kk.text(d, "火星尘埃", 800, y + 210, 32, kk.INK, a2)
    kk.text(d, "约 3 微米", 800, y + 255, 26, (110, 115, 130), a2, "reg")
    kk.text(d, "比一比大小", 540, 440, 52, kk.NAVY, vk.fade(t, 0.1), "fun")


def _norm(lam):
    i = np.argmin(abs(LAM - lam))
    return I_DUST[i] / np.trapezoid(I_DUST[i] * np.sin(TH), TH)     # 各颜色散射总量归一


_deg = np.degrees(TH)
_sel = _deg <= 15
CURVE = {lam: _norm(lam)[_sel] for lam in (450, 700)}
CURVE_X = _deg[_sel]
CURVE_MAX = max(v.max() for v in CURVE.values())
_diff = CURVE[450] - CURVE[700]
CROSS = float(CURVE_X[np.argmax(_diff < 0)])                       # 蓝光不再占优的夹角


def sc_lobe(d, t, ts):
    kk.rrect(d, 70, 360, 1010, 1240, 40, kk.WHITE, 0.92)
    kk.text(d, "离太阳多远，哪种光更亮？", 540, 440, 48, kk.NAVY, vk.fade(t, 0.1), "fun")
    x0, x1, y0, y1 = 170, 940, 540, 1010
    X = lambda a: x0 + a / 15 * (x1 - x0)
    Y = lambda v: y1 - v / CURVE_MAX * (y1 - y0)
    a = vk.fade(t, 0.3)
    shade = vk.fade(t, ts[1] + 0.8)
    d.rectangle(kk.P(x0, y0, X(CROSS), y1), fill=kk.A(kk.BLUE, 0.12 * shade))
    kk.line(d, [(x0, y0), (x0, y1), (x1, y1)], (150, 155, 170), a, 3)
    for v in (0, 5, 10, 15):
        kk.text(d, f"{v}°", X(v), y1 + 32, 26, (110, 115, 130), a, "reg")
    kk.text(d, "与太阳的夹角", 540, y1 + 80, 28, (110, 115, 130), a)
    kk.text(d, "亮度", x0 - 50, y0 + 10, 26, (110, 115, 130), a)
    for lam, col, t0, name in [(700, kk.RED, ts[0], "红光"), (450, kk.BLUE, ts[1], "蓝光")]:
        p = vk.ease((t - t0 - 0.2) / 1.2)
        n = max(2, int(len(CURVE_X) * p))
        pts = [(X(u), Y(v)) for u, v in zip(CURVE_X[:n], CURVE[lam][:n])]
        kk.line(d, pts, col, 1.0 if p > 0 else 0, 8)
        lx = 3.0 if lam == 450 else 8.5
        ly = Y(np.interp(lx, CURVE_X, CURVE[lam])) - 40
        kk.text(d, name, X(lx) + 60, ly, 34, col, vk.fade(t, t0 + 1.0))
    kk.pill(d, f"约 {CROSS:.0f}° 以内，蓝光更亮", X(CROSS / 2) + 120, y0 + 30, 30, kk.WHITE, kk.BLUE, shade,
            vk.pop(t, ts[1] + 0.8))
    kk.text(d, f"米氏散射计算：太阳正前方，蓝光强度约为红光的 {BLUE_RATIO:.1f} 倍", 540, 1170, 26, kk.INK,
            vk.fade(t, ts[1] + 1.5), "reg")


def bg_sunset(t, ts):
    return _sky_sunset.copy()


def bg_hook(t, ts):
    return _sky_hook.copy()


def sc_sunset(d, t, ts):
    blue_sun(d, 620, 1060, 24, t)
    mars_ground(d, 1080)
    rover(d, 260, 1210, 1.0)
    kk.pill(d, "按米氏散射计算出的火星日落", 540, 420, 38, kk.NAVY, kk.WHITE, vk.fade(t, 0.3))
    a = vk.fade(t, ts[0] + 1.0)
    for r in (90, 150):
        kk.circle(d, 620, 1060, r, None, a * 0.8, kk.WHITE, 3)
    kk.pill(d, "蓝色光晕", 860, 900, 32, kk.WHITE, (60, 110, 200), a, vk.pop(t, ts[0] + 1.0))
    kk.pill(d, "黄褐色天空", 250, 640, 32, kk.WHITE, RUST, vk.fade(t, ts[1]), vk.pop(t, ts[1]))


def sc_end(d, t, ts):
    a = vk.fade(t, 0.1)
    for j, (title, sub, col, x) in enumerate([("瑞利散射", "小分子 · 地球蓝天", kk.SKY, 290),
                                             ("米氏散射", "大颗粒 · 火星蓝色晚霞", MARS_DAY, 790)]):
        aj = vk.fade(t, ts[1] + 0.5 * j)
        kk.rrect(d, x - 220, 420, x + 220, 860, 34, col, aj, kk.WHITE, 5)
        if j == 0:
            for i in range(12):
                kk.circle(d, x - 150 + (i % 4) * 100, 520 + (i // 4) * 60, 6, kk.WHITE, aj)
        else:
            for i in range(4):
                d.regular_polygon((*kk.P(x - 120 + (i % 2) * 240, 530 + (i // 2) * 90), int(24 * kk.S)), 7,
                                  rotation=i * 25, fill=kk.A(DUST, aj), outline=kk.A(RUST, aj))
        kk.text(d, title, x, 740, 56, kk.WHITE, aj, "fun", stroke=6)
        kk.text(d, sub, x, 810, 28, kk.WHITE, aj, stroke=4)
    kk.text(d, "同样的阳光，不同的颜色", 540, 980, 56, kk.WHITE, a * (1 - vk.fade(t, ts[2], 0.3)), "fun", stroke=7)
    kk.next_episode(d, t, ts[2] + 0.2, "彩虹为什么是弯的？", y=1060)


DRAW = {"hook": sc_hook, "compare": sc_compare, "air": sc_air, "size": sc_size, "lobe": sc_lobe,
        "sunset": sc_sunset, "end": sc_end}
BG = {"hook": bg_hook, "sunset": bg_sunset}

if __name__ == "__main__":
    print("火星白天天空色:", MARS_DAY, " 正前方蓝/红强度比:", round(BLUE_RATIO, 2))
    kk.run(sys.modules[__name__], HERE)
