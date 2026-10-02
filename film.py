"""
薄膜干涉与发射光谱的颜色计算。

- 肥皂膜：空气中的单层水膜（n = 1.33），法向入射，计入多次反射的艾里公式：
  R = 2r²(1 − cos δ) / (1 + r⁴ − 2r² cos δ)，r = (1 − n)/(1 + n)，δ = 4πnd/λ。
  厚度趋于 0 时反射趋于 0（半波损失），即肥皂泡破裂前出现的“黑膜”。
- 焰色：以若干发射谱线或谱带（高斯线形）叠加成光谱，再按 CIE 1931 换算颜色。
"""
import numpy as np

import optics

LAM = np.arange(380.0, 781.0, 2.0)


def film_reflectance(d_nm, lam=LAM, n=1.33):
    r = (1 - n) / (1 + n)
    delta = 4 * np.pi * n * np.asarray(d_nm, dtype=float)[..., None] / lam
    return 2 * r * r * (1 - np.cos(delta)) / (1 + r ** 4 - 2 * r * r * np.cos(delta))


def film_lut(d_max=1200, step=2.0):
    """厚度 → 反射光的线性 sRGB（统一增益，保留明暗差别）。"""
    d = np.arange(0, d_max + step, step)
    spec = film_reflectance(d) * optics.sun(LAM)
    x, y, z = optics.cmf(LAM)
    XYZ = np.stack([spec @ x, spec @ y, spec @ z], -1)
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(XYZ @ M.T, 0, None)
    return d, rgb / np.percentile(rgb.max(1), 98)


def to_display(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


# 代表性发射谱线／谱带（nm, 相对强度）。焰火中锶、钡、铜的颜色主要来自 SrOH/SrCl、BaCl、CuCl 分子谱带。
EMITTERS = {
    "钠": [(589.0, 1.0), (589.6, 0.5)],
    "锂": [(670.8, 1.0)],
    "锶": [(606, 0.5), (636, 0.4), (661, 0.9), (674, 0.7), (682, 0.5)],
    "钡": [(507, 0.4), (513.9, 0.8), (524.1, 1.0), (532.1, 0.7)],
    "铜": [(420, 0.5), (428, 0.6), (435, 1.0), (443, 0.9), (452, 0.6)],
}


def emission_spectrum(lines, width=1.5):
    return sum(w * np.exp(-0.5 * ((LAM - l) / width) ** 2) for l, w in lines)


def emission_color(name):
    """发射光谱 → sRGB。单色光多超出 sRGB 色域，按惯例加入等量白光使其落入色域（保持色相）。"""
    spec = emission_spectrum(EMITTERS[name])
    x, y, z = optics.cmf(LAM)
    XYZ = np.array([spec @ x, spec @ y, spec @ z])
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    lin = M @ XYZ
    lin = lin - min(0.0, lin.min())
    lin = lin / lin.max()
    return tuple(int(round(v * 255)) for v in to_display(lin))


if __name__ == "__main__":
    d, lin = film_lut()
    rgb = (to_display(lin) * 255).astype(int)
    for t in (0, 20, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 800):
        print(f"{t:4d} nm", tuple(rgb[int(t / 2)]))
    for k in EMITTERS:
        print(k, emission_color(k))
