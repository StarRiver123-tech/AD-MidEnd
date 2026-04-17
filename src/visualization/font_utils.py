"""
字体工具模块

提供支持中文和特殊字符的文本绘制功能，
用于解决 OpenCV cv2.putText 不支持非 ASCII 字符的问题。
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List


def get_chinese_font(size: int):
    """获取支持中文的字体"""
    font_paths = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_chinese_text(canvas: np.ndarray, text: str,
                      position: Tuple[int, int],
                      color: Tuple[int, int, int],
                      font_size: int = 20):
    """在 OpenCV 图像上绘制中文文本（支持中文、特殊符号等）"""
    x, y = position
    pil_color = (color[2], color[1], color[0])  # BGR -> RGB

    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_chinese_font(font_size)
    draw.text((x, y), text, font=font, fill=pil_color)

    canvas[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_chinese_texts(canvas: np.ndarray, texts: List[str],
                       position: Tuple[int, int],
                       color: Tuple[int, int, int],
                       font_size: int, line_height: int):
    """在 OpenCV 图像上绘制多行中文文本"""
    x, y = position
    pil_color = (color[2], color[1], color[0])

    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_chinese_font(font_size)

    for i, text in enumerate(texts):
        text_y = y + i * line_height
        draw.text((x, text_y), text, font=font, fill=pil_color)

    canvas[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
