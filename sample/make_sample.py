"""生成演示样图 sample/sample.png（8 位灰度 PNG）。

设计目标（配合网页演示）：
- 图中包含平缓灰度过渡带与多组正弦纹理，Floyd–Steinberg 抖动后形成细密网点；
- 上半部（建议登记为第 1–64 行的已打样条带）网点对来自下方的误差扩散敏感，
  在条带下方不远处施加强制色时，误差会向上不可能（FS 只向后扩散）——
  因此演示的正确姿势是：在条带“内部或上方”改色会改变条带；
  另在条带之外的安全区改色则可顺利换稿。
- 尺寸 256x256，远小于 2048 上限，页面加载与续算都很快。
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

W = H = 256


def build() -> np.ndarray:
    rng = np.random.default_rng(20261126)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # 1) 自上而下的缓慢明暗过渡，保证各灰阶都有分布
    base = 96 + 72 * yy / H

    # 2) 两组正弦纹理，制造规律性网点，改色后扩散边界肉眼可辨
    texture = 28.0 * np.sin(xx / 6.0 + np.sin(yy / 11.0))
    texture += 18.0 * np.sin((xx + yy) / 9.0)

    # 3) 中部一条较亮的横带（灰度约 200）与一条较暗横带（灰度约 60）
    band_light = 46.0 * np.exp(-((yy - 150.0) ** 2) / (2 * 10.0 ** 2))
    band_dark = -46.0 * np.exp(-((yy - 200.0) ** 2) / (2 * 12.0 ** 2))

    # 4) 轻微噪声，避免完全规则的图案
    noise = rng.normal(0.0, 5.0, size=(H, W))

    gray = np.clip(base + texture + band_light + band_dark + noise, 0, 255)
    return gray.round().astype(np.uint8)


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(out_dir, "sample.png")
    img = Image.fromarray(build(), mode="L")
    img.save(path, format="PNG", optimize=True)
    print(f"样图已生成：{path} ({img.size[0]}x{img.size[1]}, mode={img.mode})")


if __name__ == "__main__":
    main()
