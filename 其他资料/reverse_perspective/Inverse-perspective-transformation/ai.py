# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 21:57:07 2025

@author: 寻木
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 假设H矩阵来自MATLAB计算


H = np.array([
[1.9194,
    6.0129,
 -901.9355],
    [0.0000,
    8.0484,
 -919.3548],
    [0.0000,
    0.0061,
    1]

], dtype=np.float32)

# 定义源点和目标点
p = np.array([[600, 570], [1200, 570], [600, 400], [1200, 400]], dtype=np.float32)
o = np.array([[1455, 570], [420, 570], [1340, 400], [550, 400]], dtype=np.float32)

# 验证数值映射
for i in range(len(p)):
    pt = np.append(p[i], 1.0)
    mapped_pt = H @ pt
    mapped_pt /= mapped_pt[2]
    print(f"源点 {p[i]} -> 映射后 [{mapped_pt[0]:.1f}, {mapped_pt[1]:.1f}] (期望目标: {o[i]})")

# 图像变换
img = cv2.imread("1.png")
warped_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("原图")
plt.subplot(122), plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)), plt.title("校正后")
plt.show()