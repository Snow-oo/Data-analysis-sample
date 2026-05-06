"""
Matplotlib 中文字体配置：优先使用宋体（Windows 为 SimSun / NSimSun）。
在 import pyplot / seaborn 绘图之前或紧接在 seaborn.set_theme 之后调用 setup_matplotlib_simsun()。
"""
from __future__ import annotations

import matplotlib


def setup_matplotlib_simsun() -> None:
    """将图表正文字体设为宋体系列，并避免负号被当成方块。"""
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "SimSun",
        "NSimSun",
        "STSong",
        "Songti SC",
        "DejaVu Serif",
        "Noto Sans CJK SC",   # Google Noto 字体
        "WenQuanYi Micro Hei", # 文泉驿微米黑
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
