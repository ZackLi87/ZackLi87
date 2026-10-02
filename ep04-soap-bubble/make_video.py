"""
第 4 期《肥皂泡为什么是彩色的？》

泡膜颜色由 film.py 按薄膜干涉（艾里公式，n = 1.33，法向入射）计算；泡泡上的彩色花纹由厚度场
逐像素查表得到：厚度场 = 重力引起的上薄下厚分布 + 缓慢旋转的扰动，并随时间整体变薄。
"""
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import film  # noqa: E402
import kidkit as kk  # noqa: E402
import vidkit as vk  # noqa: E402

NUMBER = 4
TITLE = "肥皂泡为什么是彩色的？"
COVER_T = 1.5

D_LUT, LIN_LUT = film.film_lut(1200, 2.0)
DISP_LUT = film.to_display(LIN_LUT)
NIGHT_TOP, NIGHT_LOW = (22, 40, 86), (70, 104, 170)

SCENES = [
    dict(key="hook", lines=[
        dict(show="吹一个肥皂泡，阳光下它会闪着彩色的光。"),
        dict(show="可肥皂水明明是透明的，颜色是从哪儿来的？"),
    ]),
    dict(key="layer", lines=[
        dict(show="泡泡的“皮”非常薄，只有头发丝的几百分之一。"),
        dict(show="光照上去，一部分从外面反射回来，一部分钻进去，再从里面反射回来。"),
    ]),
    dict(key="waves", lines=[
        dict(show="两束反射光叠在一起：波峰对上波峰，光就变亮；"),
        dict(show="波峰对上波谷，光就互相抵消。"),
    ]),
    dict(key="strip", lines=[
        dict(show="泡泡皮的厚度不一样，被加强的颜色也不一样，"),
        dict(show="所以泡泡上会出现一圈一圈的彩色花纹。"),
    ]),
    dict(key="thin", lines=[
        dict(show="泡泡皮里的水会慢慢往下流，上面越来越薄，颜色也跟着变；"),
        dict(show="薄到几乎看不见颜色的时候，泡泡就快要破啦！"),
    ]),
    dict(key="end", lines=[
        dict(show="这种现象叫光的干涉。"),
        dict(show="水面上的油膜、有些甲虫闪亮的外壳，也是这个道理。"),
        dict(show="下一期，我们来看看：烟花为什么五颜六色？"),
    ]),
]


# ---------------------------------------------------------------- 泡泡渲染
def thickness_field(n, t, mean, swirl=110.0):
    """n×n 网格上的膜厚（nm），圆盘外为 NaN。"""
    v, u = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    rr = np.hypot(u, v)
    base = mean * (0.25 + 1.1 * (v + 1) / 2)                          # 上薄下厚
    vv = v + 0.18 * np.sin(2.2 * u + 0.6 * t) + 0.1 * np.sin(4.1 * u - 0.9 * t)   # 被扰动的水平条带
    wob = swirl * (0.8 * np.sin(5.5 * vv + 0.7 * t) + 0.5 * np.sin(3.1 * u + 2.2 * v + 0.6 * t)
                   * np.sin(2.4 * v - 1.3 * u - 0.4 * t))
    d = np.clip(base + wob * np.clip(mean / 500, 0.15, 1), 0, 1199)
    d[rr > 1] = np.nan
    return d, rr


def draw_bubble(d, cx, cy, R, t, mean, a=1.0, swirl=110.0):
    """在画布上叠加由计算得到的泡膜反射色（加色混合），再画高光与轮廓。"""
    if a <= 0.01 or R < 4:
        return
    S = kk.S
    n = int(2 * R * S)
    th, rr = thickness_field(n, t, mean, swirl)
    mask = ~np.isnan(th)
    idx = np.clip(np.nan_to_num(th) / 2.0, 0, len(D_LUT) - 1).astype(int)
    col = DISP_LUT[idx] * 255 * 0.9
    edge = np.clip((1 - rr) * 25, 0, 1)[..., None]                    # 边缘柔化
    x0, y0 = int((cx - R) * S), int((cy - R) * S)
    img = d._image
    box = (x0, y0, x0 + n, y0 + n)
    region = np.asarray(img.crop(box), dtype=float)
    out = region * (1 - 0.15 * edge * a) + col * edge * mask[..., None] * a * 0.85
    img.paste(Image.fromarray(out.clip(0, 255).astype(np.uint8)), box[:2])
    kk.circle(d, cx, cy, R, None, a * 0.45, kk.WHITE, 3)
    d.ellipse(kk.P(cx - R * 0.55, cy - R * 0.6, cx - R * 0.15, cy - R * 0.38), fill=kk.A(kk.WHITE, a * 0.55))


def bg_night(t, ts):
    return kk.gradient(NIGHT_TOP, NIGHT_LOW)


# ---------------------------------------------------------------- 各场景
def sc_hook(d, t, ts):
    for i, (x, y, r, ph) in enumerate([(290, 900, 150, 0), (730, 820, 210, 1.3), (620, 1180, 100, 2.1)]):
        yy = y - 18 * t + 10 * math.sin(t * 1.2 + ph)
        draw_bubble(d, x, yy, r, t + ph, 520 - 60 * i)
    kk.text(d, "肥皂泡为什么", 540, 330, 120, kk.WHITE, 1.0, "fun", stroke=10)
    kk.text(d, "是彩色的？", 540, 470, 120, kk.WHITE, 1.0, "fun", stroke=10)


def sc_layer(d, t, ts):
    a = vk.fade(t, 0.1)
    draw_bubble(d, 280, 520, 130, t, 450, a)
    kk.line(d, [(380, 470), (560, 400)], kk.WHITE, a, 3)
    kk.pill(d, "放大泡泡皮", 700, 380, 32, kk.NAVY, kk.WHITE, a)
    y1, y2 = 820, 960                                                   # 膜的上下表面（示意，厚度放大）
    d.rectangle(kk.P(80, y1, 1000, y2), fill=kk.A((150, 200, 255), 0.35 * a))
    kk.line(d, [(80, y1), (1000, y1)], kk.WHITE, a, 4)
    kk.line(d, [(80, y2), (1000, y2)], kk.WHITE, a, 4)
    kk.text(d, "约 0.0001–0.001 毫米", 840, y2 + 40, 28, kk.WHITE, vk.fade(t, ts[0] + 1.0), stroke=4)
    p = vk.ease((t - ts[1] - 0.2) / 2.4)
    if p > 0:
        A, B = (230, 560), (430, y1)
        kk.line(d, [A, (A[0] + (B[0] - A[0]) * min(1, p * 3), A[1] + (B[1] - A[1]) * min(1, p * 3))], kk.SUNY, 1, 7)
    if p > 0.33:
        q = (p - 0.33) / 0.67
        kk.line(d, [(430, y1), (430 + 200 * min(1, q * 2), y1 - 260 * min(1, q * 2))], kk.SUNY, 1, 6)
        C = (500, y2)
        kk.line(d, [(430, y1), (430 + 70 * min(1, q * 3), y1 + 140 * min(1, q * 3))], (255, 230, 150), 1, 5)
        if q > 0.33:
            r2 = min(1, (q - 0.33) * 3)
            kk.line(d, [C, (C[0] + 70 * r2, C[1] - 140 * r2)], (255, 230, 150), 1, 5)
        if q > 0.66:
            r3 = (q - 0.66) / 0.34
            kk.line(d, [(570, y1), (570 + 200 * r3, y1 - 260 * r3)], (255, 190, 80), 1, 6)
    kk.pill(d, "① 外面反射", 650, 600, 30, kk.WHITE, (200, 140, 20), vk.fade(t, ts[1] + 1.2), vk.pop(t, ts[1] + 1.2))
    kk.pill(d, "② 里面反射", 870, 700, 30, kk.WHITE, (220, 110, 40), vk.fade(t, ts[1] + 2.4), vk.pop(t, ts[1] + 2.4))


def two_waves(d, y, phase, col1, col2, t, a, label):
    xs = np.arange(120, 961, 6)
    w1 = [(x, y - 120 + 40 * math.sin(x / 45 - t * 3)) for x in xs]
    w2 = [(x, y - 30 + 40 * math.sin(x / 45 - t * 3 + phase)) for x in xs]
    s = [(x, y + 110 + 40 * (math.sin(x / 45 - t * 3) + math.sin(x / 45 - t * 3 + phase))) for x in xs]
    kk.line(d, w1, col1, a, 6)
    kk.line(d, w2, col2, a, 6)
    kk.text(d, "+", 70, y - 75, 50, kk.WHITE, a, stroke=4)
    kk.text(d, "=", 70, y + 110, 50, kk.WHITE, a, stroke=4)
    kk.line(d, s, kk.WHITE, a, 9)
    kk.pill(d, label, 540, y + 230, 34, kk.NAVY, kk.WHITE, a)


def sc_waves(d, t, ts):
    a1 = vk.fade(t, ts[0]) * (1 - vk.fade(t, ts[1], 0.4))
    a2 = vk.fade(t, ts[1] + 0.3)
    two_waves(d, 760, 0.0, kk.SUNY, (255, 190, 80), t, a1, "变亮")
    two_waves(d, 760, math.pi, kk.SUNY, (255, 190, 80), t, a2, "抵消")


def sc_strip(d, t, ts):
    x0, x1, y0, y1 = 100, 980, 980, 1100
    a = vk.fade(t, 0.2)
    kk.rrect(d, x0 - 20, 330, x1 + 20, 1220, 36, (10, 16, 40), a * 0.6)
    kk.text(d, "按光的干涉算出的颜色", 540, 400, 48, kk.WHITE, a, "fun", stroke=5)
    dmax = 800
    for k in range(0, x1 - x0, 2):
        dd = k / (x1 - x0) * dmax
        c = tuple(int(v) for v in (DISP_LUT[int(dd / 2)] * 255))
        d.rectangle(kk.P(x0 + k, y0, x0 + k + 2.5, y1), fill=kk.A(c, a))
    for v in (0, 200, 400, 600, 800):
        kk.text(d, f"{v}", x0 + v / dmax * (x1 - x0), y1 + 30, 26, kk.WHITE, a, "reg", stroke=3)
    kk.text(d, "泡泡皮的厚度（纳米）", 540, y1 + 75, 28, kk.WHITE, a, stroke=3)
    pos = 30 + 760 * (0.5 - 0.5 * math.cos(max(0, t - 0.6) * 0.55))
    mx = x0 + pos / dmax * (x1 - x0)
    kk.line(d, [(mx, y0 - 30), (mx, y1 + 5)], kk.WHITE, a, 5)
    c = tuple(int(v) for v in (DISP_LUT[int(pos / 2)] * 255))
    kk.circle(d, 540, 680, 170, c, a, kk.WHITE, 6)
    kk.pill(d, f"{pos:.0f} 纳米", 540, 900, 34, kk.NAVY, kk.WHITE, a)


def sc_thin(d, t, ts):
    span = ts[1] + 2.5
    mean = 600 - 560 * vk.ease(t / span)
    pop_t = ts[1] + 2.8
    if t < pop_t:
        draw_bubble(d, 540, 760, 330, t, mean, 1.0, swirl=110 * max(0.25, mean / 600))
        a = vk.fade(t, ts[1] + 0.8)
        kk.pill(d, "顶部几乎透明：快破了！", 540, 360, 34, kk.NAVY, kk.WHITE, a, vk.pop(t, ts[1] + 0.8))
        kk.text(d, f"平均厚度约 {mean:.0f} 纳米", 540, 1150, 32, kk.WHITE, 1.0, stroke=4)
    else:
        k = (t - pop_t) / 0.8
        rng = np.random.default_rng(1)
        for i in range(40):
            th = rng.uniform(0, 2 * math.pi)
            r = 330 * (0.8 + 0.6 * k) * rng.uniform(0.7, 1.1)
            kk.circle(d, 540 + r * math.cos(th), 760 + r * math.sin(th) + 300 * k * k, 7, (200, 230, 255),
                      max(0, 1 - k))
        kk.text(d, "啪！", 540, 760, 160 * vk.pop(t, pop_t, 0.3), kk.WHITE, max(0, 1 - (k - 1)), "fun", stroke=10)


def sc_end(d, t, ts):
    a = vk.fade(t, 0.1)
    kk.text(d, "光的干涉", 540, 430, 120, kk.WHITE, a, "fun", stroke=10)
    for i, (x, y, r) in enumerate([(300, 760, 120), (780, 720, 150)]):
        draw_bubble(d, x, y - 10 * t, r, t + i, 450 - 80 * i, vk.fade(t, ts[1]) * (1 - vk.fade(t, ts[2], 0.4)))
    kk.next_episode(d, t, ts[2] + 0.2, "烟花为什么五颜六色？", y=760)


DRAW = {"hook": sc_hook, "layer": sc_layer, "waves": sc_waves, "strip": sc_strip, "thin": sc_thin, "end": sc_end}
BG = {k: bg_night for k in DRAW}

if __name__ == "__main__":
    kk.run(sys.modules[__name__], HERE)
