import matplotlib.pyplot as plt
import numpy as np
import cv2
from math import sqrt,pow

def blpf(img, r):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    transfor_matrix = np.zeros(img.shape)
    M = transfor_matrix.shape[0]
    N = transfor_matrix.shape[1]
    for u in range(M):
        for v in range(N):
            D = sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            transfor_matrix[u, v] = 1 / (1 + pow(D / r, 8))
    new = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * transfor_matrix)))
    return new