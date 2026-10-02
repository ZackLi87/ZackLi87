"""
第 5 期《烟花为什么五颜六色？》（第一阶段收官）

各元素的焰色由 film.py 中的发射谱线／谱带按 CIE 1931 计算得到；烟花为粒子模拟
（初速度随机、重力与空气阻力、寿命衰减）。
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import film  # noqa: E402
import kidkit as kk  # noqa: E402
import vidkit as vk  # noqa: E402

NUMBER = 5
TITLE = "烟花为什么五颜六色？"
COVER_T = 2.2

ELEM = ["钠", "锶", "钡", "铜"]
COL = {k: film.emission_color(k) for k in film.EMITTERS}
COLOR_NAME = {"钠": "黄", "锶": "红", "钡": "绿", "铜": "蓝"}
NIGHT_TOP, NIGHT_LOW = (8, 10, 32), (38, 34, 78)

SCENES = [
    dict(key="hook", lines=[
        dict(show="嘭！夜空中绽放出红的、绿的、蓝的烟花。"),
        dict(show="烟花为什么会有这么多颜色？"),
    ]),
    dict(key="metal", lines=[
        dict(show="秘密在烟花里装的金属元素。"),
        dict(show="钠会发出黄光，锶是红光，钡是绿光，铜是蓝光。"),
    ]),
    dict(key="atom", lines=[
        dict(show="烟花里温度很高，电子吸收能量，跳到外面的轨道上；"),
        dict(show="它跳回来的时候，就把能量变成一束光放出来。"),
    ]),
    dict(key="spectrum", lines=[
        dict(show="每种元素放出的光，都有自己固定的颜色，"),
        dict(show="就像每个人都有自己的指纹一样。"),
    ]),
    dict(key="flame", lines=[
        dict(show="化学课上的焰色反应实验，也是这个原理："),
        dict(show="把不同的金属盐放进火焰里，火焰就会变成不同的颜色。"),
    ]),
    dict(key="end", lines=[
        dict(show="所以，烟花的颜色，其实是元素在发光。"),
        dict(show="好奇心实验室的前五期就到这里啦！"),
        dict(show="你还想知道哪些为什么？在评论区告诉我，下一期就做你的问题！"),
    ]),
]


# ---------------------------------------------------------------- 烟花粒子模拟
def make_show(duration, seed, plan):
    """plan: [(爆炸时刻, x, y, 元素)]。返回各粒子的轨迹函数所需参数。"""
    rng = np.random.default_rng(seed)
    shells = []
    for t0, x, y, e in plan:
        n = 90
        th = rng.uniform(0, 2 * math.pi, n)
        sp = rng.normal(330, 40, n)
        shells.append(dict(t0=t0, x=x, y=y, col=COL[e], vx=sp * np.cos(th), vy=sp * np.sin(th),
                           life=rng.uniform(1.4, 2.0, n), x0=rng.uniform(x - 60, x + 60)))
    return shells


def pos(sh, tau):
    k = 1.6                                                             # 空气阻力系数
    f = (1 - np.exp(-k * tau)) / k
    x = sh["x"] + sh["vx"] * f
    y = sh["y"] + sh["vy"] * f + 0.5 * 160 * tau ** 2
    return x, y


def draw_show(d, shells, t, a=1.0):
    for sh in shells:
        tau = t - sh["t0"]
        if -0.8 < tau < 0:                                                # 升空
            u = (tau + 0.8) / 0.8
            yy = 1500 - (1500 - sh["y"]) * (1 - (1 - u) ** 2)
            xx = sh["x0"] + (sh["x"] - sh["x0"]) * u
            kk.line(d, [(xx, yy), (xx, yy + 60)], (255, 220, 160), a * 0.6, 4)
            kk.circle(d, xx, yy, 6, (255, 240, 200), a)
        elif tau >= 0:
            x, y = pos(sh, tau)
            xp, yp = pos(sh, max(0.0, tau - 0.12))
            fadev = np.clip(1 - tau / sh["life"], 0, 1)
            for i in range(len(x)):
                if fadev[i] <= 0.02:
                    continue
                al = a * fadev[i]
                kk.line(d, [(xp[i], yp[i]), (x[i], y[i])], sh["col"], al * 0.6, 4)
                kk.circle(d, x[i], y[i], 13, sh["col"], al * 0.4)
                kk.circle(d, x[i], y[i], 7, sh["col"], al)
                kk.circle(d, x[i], y[i], 3.5, (255, 255, 255), al * 0.9)
            if tau < 0.25:
                kk.circle(d, sh["x"], sh["y"], 120 * tau / 0.25, sh["col"], a * (1 - tau / 0.25) * 0.5)


SHOWS = {}


def setup(tl):
    SHOWS["hook"] = make_show(10, 1, [(0.3, 300, 560, "锶"), (0.9, 760, 470, "钡"), (1.6, 540, 760, "铜"),
                                      (2.6, 330, 430, "钠"), (3.3, 770, 700, "锶"), (4.2, 520, 520, "钡"),
                                      (5.2, 300, 760, "铜"), (6.0, 760, 480, "钠")])
    SHOWS["end"] = make_show(16, 2, [(0.2 + 1.1 * i, [300, 760, 540, 420, 680][i % 5],
                                      [560, 480, 720, 420, 640][i % 5], ELEM[i % 4]) for i in range(14)])


def bg_night(t, ts):
    return kk.gradient(NIGHT_TOP, NIGHT_LOW)


# ---------------------------------------------------------------- 各场景
def sc_hook(d, t, ts):
    draw_show(d, SHOWS["hook"], t)
    kk.text(d, "烟花为什么", 540, 1060, 120, kk.WHITE, 1.0, "fun", stroke=10)
    kk.text(d, "五颜六色？", 540, 1200, 120, kk.WHITE, 1.0, "fun", stroke=10)


def sc_metal(d, t, ts):
    a = vk.fade(t, 0.1)
    kk.text(d, "烟花里的“颜色配方”", 540, 400, 56, kk.WHITE, a, "fun", stroke=6)
    for i, e in enumerate(ELEM):
        x, y = 290 + (i % 2) * 500, 640 + (i // 2) * 360
        t0 = ts[1] + 0.35 + 0.9 * i
        s = vk.pop(t, t0)
        kk.circle(d, x, y, 120 * s, COL[e], 0.25)
        kk.circle(d, x, y, 95 * s, COL[e], 1.0, kk.WHITE, 4)
        kk.text(d, e, x, y - 5, 80 * s, kk.WHITE, 1.0, "fun", stroke=6)
        kk.pill(d, f"{COLOR_NAME[e]}光", x, y + 150, 34, kk.WHITE, (30, 34, 70), vk.fade(t, t0), s)


def sc_atom(d, t, ts):
    cx, cy = 540, 760
    e = ELEM[int(max(0, t - ts[1]) / 1.6) % 4] if t > ts[1] else "钠"
    col = COL[e]
    kk.circle(d, cx, cy, 50, (240, 110, 90), 1.0, kk.WHITE, 3)
    for r in (160, 300):
        kk.circle(d, cx, cy, r, None, 0.6, kk.WHITE, 3)
    heat = vk.fade(t, ts[0] + 0.6)
    cyc = (t - ts[1]) % 1.6 if t > ts[1] else 0
    up = vk.ease((t - ts[0] - 1.2) / 0.8) if t < ts[1] else (1 - vk.ease(cyc / 0.35) if cyc < 0.8 else vk.ease((cyc - 0.8) / 0.6))
    r = 160 + 140 * up
    ang = t * 1.3
    ex, ey = cx + r * math.cos(ang), cy + r * math.sin(ang)
    kk.circle(d, ex, ey, 22, (120, 200, 255), 1.0, kk.WHITE, 3)
    for k in range(5):                                               # 火焰提供能量
        fx = 160 + k * 190
        fh = 60 + 20 * math.sin(t * 8 + k)
        d.polygon([tuple(kk.P(*p)) for p in flame_shape(fx, 1200, fh * 2, 40, t + k)],
                  fill=kk.A((255, 140, 40), heat * 0.8))
        d.polygon([tuple(kk.P(*p)) for p in flame_shape(fx, 1200, fh, 18, t + k + 0.5)],
                  fill=kk.A((255, 220, 120), heat * 0.8))
    kk.pill(d, "加热：电子跳上去", 540, 400, 34, kk.NAVY, kk.WHITE, vk.fade(t, ts[0] + 1.0))
    if t > ts[1] and cyc < 0.9:                                      # 落回时放出光子
        u = cyc / 0.9
        pts = [(ex + (40 + 500 * u) * math.cos(ang) + 18 * math.sin(s / 10) * -math.sin(ang),
                ey + (40 + 500 * u) * math.sin(ang) + 18 * math.sin(s / 10) * math.cos(ang))
               for s in np.linspace(0, 160, 40)]
        kk.line(d, pts, col, 1 - u * 0.5, 8)
        kk.pill(d, f"跳回来：放出{COLOR_NAME[e]}光（{e}）", 540, 470, 34, kk.WHITE, (30, 34, 70), 1.0)


def sc_spectrum(d, t, ts):
    a = vk.fade(t, 0.1)
    kk.rrect(d, 60, 330, 1020, 1250, 36, (10, 12, 30), a * 0.8)
    kk.text(d, "元素的“光谱指纹”", 540, 400, 52, kk.WHITE, a, "fun", stroke=5)
    x0, x1 = 230, 860
    X = lambda l: x0 + (l - 400) / 300 * (x1 - x0)
    for i, e in enumerate(ELEM):
        y = 530 + i * 170
        ai = vk.fade(t, 0.4 + 0.5 * i)
        kk.text(d, e, 140, y, 56, kk.WHITE, ai, "fun", stroke=4)
        kk.rrect(d, x0, y - 45, x1, y + 45, 10, (24, 26, 50), ai)
        for l, w in film.EMITTERS[e]:
            if 400 <= l <= 700:
                kk.line(d, [(X(l), y - 40), (X(l), y + 40)], vk.lerp_col((60, 60, 80), COL[e], w), ai, 5)
        kk.text(d, "=", 900, y, 44, kk.WHITE, ai, stroke=3)
        kk.circle(d, 960, y, 34, COL[e], ai, kk.WHITE, 3)
    for v, s in ((400, "400"), (550, "550"), (700, "700 纳米")):
        kk.text(d, s, X(v), 1210, 24, kk.WHITE, a, "reg")
    kk.pill(d, "每条亮线的位置都是固定的", 540, 1150, 30, kk.NAVY, kk.WHITE, vk.fade(t, ts[1]), vk.pop(t, ts[1]))


def flame_shape(cx, base, h, w, t):
    pts = []
    for k in range(25):
        u = k / 24
        x = cx - w * math.sin(math.pi * u) * (1 - 0.3 * u)
        pts.append((x + 8 * math.sin(t * 9 + u * 6), base - h * u))
    for k in range(24, -1, -1):
        u = k / 24
        x = cx + w * math.sin(math.pi * u) * (1 - 0.3 * u)
        pts.append((x + 8 * math.sin(t * 9 + u * 6 + 1), base - h * u))
    return pts


def sc_flame(d, t, ts):
    cx, base = 540, 1050
    k = int(max(0, t - ts[1]) / 1.4) % 4 if t > ts[1] else -1
    col = COL[ELEM[k]] if k >= 0 else (90, 140, 255)
    kk.rrect(d, cx - 45, base, cx + 45, base + 170, 10, (150, 160, 180))           # 酒精喷灯（示意）
    kk.rrect(d, cx - 120, base + 160, cx + 120, base + 190, 10, (120, 130, 150))
    outer = flame_shape(cx, base, 430, 110, t)
    inner = flame_shape(cx, base, 220, 45, t + 0.5)
    d.polygon([tuple(kk.P(*p)) for p in outer], fill=kk.A(col, 0.75))
    d.polygon([tuple(kk.P(*p)) for p in inner], fill=kk.A(vk.lerp_col(col, (255, 255, 255), 0.5), 0.8))
    if k >= 0:
        kk.pill(d, f"{ELEM[k]}盐 → {COLOR_NAME[ELEM[k]]}色火焰", cx, 420, 40, kk.WHITE, (30, 34, 70), 1.0,
                vk.pop(t, ts[1] + 1.4 * k))
    else:
        kk.pill(d, "焰色反应", cx, 420, 44, kk.NAVY, kk.WHITE, vk.fade(t, 0.2))
    kk.text(d, "实验请在老师指导下进行", 540, 1290, 26, kk.WHITE, 0.85, "reg", stroke=3)


def sc_end(d, t, ts):
    draw_show(d, SHOWS["end"], t)
    a = vk.fade(t, ts[1])
    kk.text(d, "前五期完结", 540, 1000, 96, kk.WHITE, a, "fun", stroke=9)
    kk.pill(d, "你还想知道什么？评论区告诉我", 540, 1140, 40, kk.NAVY, kk.SUNY, vk.fade(t, ts[2] + 0.3),
            vk.pop(t, ts[2] + 0.3))
    kk.text(d, "燃放烟花请遵守当地规定，并由成人在安全场所进行", 540, 1290, 24, kk.WHITE, 0.85, "reg", stroke=3)


DRAW = {"hook": sc_hook, "metal": sc_metal, "atom": sc_atom, "spectrum": sc_spectrum, "flame": sc_flame, "end": sc_end}
BG = {k: bg_night for k in DRAW}

if __name__ == "__main__":
    print("焰色:", COL)
    kk.run(sys.modules[__name__], HERE)
