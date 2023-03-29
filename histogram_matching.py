import numpy as np

def histogram_matching(image, target):
    ## TODO 实现直方图匹配函数，输入源图像和参考图像，返回匹配以后的图像
    result = np.zeros_like(target)
    hist_img, m = np.histogram(image[:, :], 256)
    hist_target, n = np.histogram(target[:, :], 256)
    cdf_img = np.cumsum(hist_img)
    cdf_target = np.cumsum(hist_target)
    for j in range(256):
        tmp = abs(cdf_img[j] - cdf_target)
        tmp = tmp.tolist()
        index = tmp.index(min(tmp))
        result[:, :][image[:, :] == j] = index
    return result