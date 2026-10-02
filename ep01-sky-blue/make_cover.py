"""生成抖音所需的竖版（3:4）与横版（4:3）封面。复用 make_video.py 的绘图函数与计算颜色。"""
import argparse
import os
import sys

from PIL import ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_video as mv  # noqa: E402


def cover(w, h, out):
    mv.W, mv.H = w, h
    img = mv.gradient(mv.SKY_TOP, mv.SKY_LOW)
    d = ImageDraw.Draw(img, "RGBA")
    portrait = h > w
    if portrait:
        mv.cloud(d, 170, 330, 62)
        mv.cloud(d, 860, 560, 48, 0.9)
        mv.cute_sun(d, 840, 250, 92, 0.3)
        d.ellipse(mv.P(-420, 1100, 1500, 2300), fill=mv.A(mv.GRASS))
        mv.kid(d, 250, 1170, 1.0, scale=1.25)
        mv.text(d, "天为什么", 540, 620, 150, kind="fun", stroke=10)
        mv.text(d, "是蓝色的？", 540, 800, 150, kind="fun", stroke=10)
        mv.pill(d, "用物理公式算给孩子看", 640, 1010, 44, mv.NAVY, mv.WHITE)
    else:
        mv.cloud(d, 180, 230, 60)
        mv.cloud(d, 1250, 560, 50, 0.9)
        mv.cute_sun(d, 1230, 230, 100, 0.3)
        d.ellipse(mv.P(-300, 870, 1740, 1900), fill=mv.A(mv.GRASS))
        mv.kid(d, 230, 1000, 1.0, scale=1.15)
        mv.text(d, "天为什么是蓝色的？", 720, 450, 140, kind="fun", stroke=10)
        mv.pill(d, "用物理公式算给孩子看", 720, 660, 46, mv.NAVY, mv.WHITE)
    img.reduce(mv.S).save(out, quality=92)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(HERE, "assets"))
    mv.ARGS = ap.parse_args()
    cover(1080, 1440, os.path.join(HERE, "cover_3x4.jpg"))
    cover(1440, 1080, os.path.join(HERE, "cover_4x3.jpg"))
    print("ok")
