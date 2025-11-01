import cv2
import numpy as np
import matplotlib.pyplot as plt


# img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('letterA.png', cv2.IMREAD_GRAYSCALE)
# img = cv2.GaussianBlur(img,(3,3),0.5)
img = np.float32(img)

filterX = np.array([[1,0,-1],
 [2,0,-2],
 [1,0,-1]])
filterY = np.array([[1,2,1],
 [0,0,0],
 [-1,-2,-1]])
Ix = cv2.filter2D(img,-1,filterX)
Iy = cv2.filter2D(img,-1,filterY)

Ixy = Ix * Iy
Ixx = Ix * Ix
Iyy = Iy * Iy
Ixy = cv2.GaussianBlur(Ixy,(3,3),0.5)
Ixx = cv2.GaussianBlur(Ixx,(3,3),0.5)
Iyy = cv2.GaussianBlur(Iyy,(3,3),0.5)

det_M = Ixx * Iyy - Ixy * Ixy
trace_M = Ixx + Iyy
k = 0.04

R = det_M - k * trace_M * trace_M

max_R = np.max(R)
threshold = max_R * 0.01
corners = np.zeros_like(R)
corners[R > threshold] = 1

# 非极大值抑制（可选，用于减少相邻角点）
def non_maximum_suppression(corners, R, window_size=5):
    """
    非极大值抑制，保留局部最大值点
    """
    result = np.zeros_like(corners)
    padded_R = np.pad(R, window_size//2, mode='constant')
    
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if corners[i, j] == 1:
                # 检查是否是局部最大值
                window = padded_R[i:i+window_size, j:j+window_size]
                if R[i, j] == np.max(window):
                    result[i, j] = 1
    
    return result

# 应用非极大值抑制
corners_nms = non_maximum_suppression(corners, R, window_size=9)



