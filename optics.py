"""
天空颜色的物理计算（单次瑞利散射近似）。

- 太阳光谱：5778 K 黑体辐射（近似）
- 瑞利光学厚度：tau(λ) = 0.008569 λ^-4 (1 + 0.0113 λ^-2 + 0.00013 λ^-4)，λ 单位 μm
- 人眼响应：CIE 1931 2° 标准观察者，采用 Wyman、Sloan 与 Shirley（2013）的多高斯解析拟合
- 色彩空间：XYZ → 线性 sRGB → sRGB（D65）
"""
import numpy as np

LAM = np.arange(380.0, 781.0, 1.0)              # 波长，nm


def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def cmf(lam=LAM):
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return x, y, z


def sun(lam=LAM, T=5778.0):
    l = lam * 1e-9
    return 1.0 / (l ** 5 * (np.exp(1.4388e-2 / (l * T)) - 1))


def tau(lam=LAM):
    u = lam / 1000.0
    return 0.008569 * u ** -4 * (1 + 0.0113 * u ** -2 + 0.00013 * u ** -4)


def to_srgb(spec, lam=LAM, bright=1.0):
    """光谱 → sRGB（0–255），按最大分量归一化后乘以亮度。"""
    x, y, z = cmf(lam)
    X, Y, Z = (spec * x).sum(), (spec * y).sum(), (spec * z).sum()
    rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415],
                    [0.0557, -0.2040, 1.0570]]) @ np.array([X, Y, Z])
    rgb = np.clip(rgb, 0, None)
    rgb = rgb / (rgb.max() + 1e-12) * bright
    g = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return tuple(int(round(v)) for v in np.clip(g, 0, 1) * 255)


def airmass(zenith_deg):
    """Kasten–Young 大气质量公式。"""
    z = np.radians(zenith_deg)
    return 1.0 / (np.cos(z) + 0.50572 * (96.07995 - zenith_deg) ** -1.6364)


def sky_color(m=1.0):
    """散射天光：λ^-4 散射，并计入路径上的衰减。"""
    return to_srgb(sun() * tau() * np.exp(-tau() * m))


def sun_color(m=1.0):
    """穿过 m 倍大气后的直射阳光颜色。"""
    return to_srgb(sun() * np.exp(-tau() * m))


def wavelength_rgb(lam_nm):
    """单色光的显示颜色（Bruton 近似；单色光多在 sRGB 色域之外，仅作示意）。"""
    l = float(lam_nm)
    if l < 440:
        r, g, b = -(l - 440) / 60, 0.0, 1.0
    elif l < 490:
        r, g, b = 0.0, (l - 440) / 50, 1.0
    elif l < 510:
        r, g, b = 0.0, 1.0, -(l - 510) / 20
    elif l < 580:
        r, g, b = (l - 510) / 70, 1.0, 0.0
    elif l < 645:
        r, g, b = 1.0, -(l - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    f = 0.3 + 0.7 * (l - 380) / 40 if l < 420 else (0.3 + 0.7 * (780 - l) / 80 if l > 700 else 1.0)
    return tuple(int(round(255 * (c * f) ** 0.8)) for c in (r, g, b))


SCATTER_RATIO = (700.0 / 450.0) ** 4             # 蓝光（450 nm）与红光（700 nm）的散射强度之比

if __name__ == "__main__":
    print("天空（正午）:", sky_color(1.0))
    for zd in (0, 60, 80, 85, 88, 90):
        m = airmass(zd)
        print(f"天顶角 {zd:2d}°  大气质量 {m:5.1f}  太阳 {sun_color(m)}  天光 {sky_color(m)}")
    print("蓝/红散射比:", round(SCATTER_RATIO, 2))
    print([wavelength_rgb(l) for l in (700, 620, 580, 530, 490, 450, 410)])
