"""
主虹的几何光学计算：光线进入水滴折射一次、内反射一次、射出时再折射一次。
水的折射率采用 Daimon & Masumura（2007，20 °C）色散公式。
"""
import numpy as np


def n_water(lam_nm):
    l2 = (np.asarray(lam_nm, dtype=float) / 1000.0) ** 2
    B = [5.684027565e-1, 1.726177391e-1, 2.086189578e-2, 1.130748688e-1]
    C = [5.101829712e-3, 1.821153936e-2, 2.620722293e-2, 1.069792721e1]
    return np.sqrt(1 + sum(b * l2 / (l2 - c) for b, c in zip(B, C)))


def exit_angle(b, lam_nm):
    """入射高度 b（以水滴半径为 1）→ 出射光与反向入射光的夹角（度），即观察者看到的角度。"""
    i = np.arcsin(b)
    r = np.arcsin(np.sin(i) / n_water(lam_nm))
    return np.degrees(4 * r - 2 * i)


def rainbow_angle(lam_nm):
    b = np.linspace(0, 1, 200001)
    return exit_angle(b, lam_nm).max()


if __name__ == "__main__":
    for l in (400, 450, 500, 550, 600, 650, 700):
        print(l, "n =", round(float(n_water(l)), 4), " 虹角 =", round(rainbow_angle(l), 2), "°")
