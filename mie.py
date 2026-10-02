"""
球形颗粒的米氏散射（Bohren & Huffman, 1983，BHMIE 算法的 numpy 实现）。
"""
import numpy as np


def mie(m, x, mu):
    """返回散射振幅 S1、S2（对各 cosθ）以及 Qsca、Qext。m 为复折射率，x = 2πr/λ。"""
    mu = np.asarray(mu, dtype=float)
    nstop = int(x + 4 * x ** (1 / 3) + 2)
    y = m * x
    nmx = int(max(nstop, abs(y)) + 15)
    D = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):                       # 对数导数，向下递推
        D[n - 1] = n / y - 1 / (D[n] + n / y)
    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = psi1 - 1j * chi1
    pi0, pi1 = np.zeros_like(mu), np.ones_like(mu)
    S1 = np.zeros_like(mu, dtype=complex)
    S2 = np.zeros_like(mu, dtype=complex)
    qsca = qext = 0.0
    for n in range(1, nstop + 1):
        fn = (2 * n + 1) / (n * (n + 1))
        psi = (2 * n - 1) * psi1 / x - psi0
        chi = (2 * n - 1) * chi1 / x - chi0
        xi = psi - 1j * chi
        da, db = D[n] / m + n / x, m * D[n] + n / x
        a = (da * psi - psi1) / (da * xi - xi1)
        b = (db * psi - psi1) / (db * xi - xi1)
        qsca += (2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)
        qext += (2 * n + 1) * (a + b).real
        tau = n * mu * pi1 - (n + 1) * pi0
        S1 += fn * (a * pi1 + b * tau)
        S2 += fn * (a * tau + b * pi1)
        psi0, psi1, chi0, chi1 = psi1, psi, chi1, chi
        xi1 = psi1 - 1j * chi1
        pi0, pi1 = pi1, ((2 * n + 1) * mu * pi1 - (n + 1) * pi0) / n
    return S1, S2, 2 * qsca / x ** 2, 2 * qext / x ** 2
