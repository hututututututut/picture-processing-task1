import numpy as np 
import cv2
import matplotlib.pyplot as plt
def fourier_low_pass(image, r):
    """
    :param image: 输入图像
    :param r: 低通范围半径
    :return: 处理后的图像
    """
    ## TODO 实现低通滤波
    f = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f)
    # 初始化
    m = f_shift.shape[0]
    n = f_shift.shape[1]
    h1 = np.zeros((m, n))
    x0 = np.floor(m / 2)
    y0 = np.floor(n / 2)
    for i in range(m):
        for j in range(n):
            d = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
            if d <= r:
                h1[i][j] = 1
    glpf = np.multiply(f_shift, h1)
    new = np.fft.ifftshift(glpf)
    result = np.abs(np.fft.ifft2(new))
    return result,f_shift,h1,glpf


   


