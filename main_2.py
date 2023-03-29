import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from histogram_matching import histogram_matching
import cv2
from fourier_low_pass import fourier_low_pass
from Filters import MediumFilter,BilateralFilter
from blpf import blpf

plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.rcParams['figure.dpi'] = 300
# load image加载图像
img = Image.open('lung.png').convert('L')
img = np.array(img)
# 显示原图像
plt.subplot(231)  # 23:使输出的图像排版为2行3列，1:在2行3列中的第一行第一列显示
plt.imshow(img, cmap="gray")
plt.title("input")  # 设置图片的标题
plt.axis('off')  # 关闭坐标轴显示
# 加载目标图像
target = Image.open('target.png').convert('L')
target = np.array(target)
# 显示目标图像
plt.subplot(232)  # 23:使输出的图像排版为2行3列，1:在2行3列中的第一行第二列显示
plt.imshow(target, cmap="gray")
plt.title("target")  # 设置图片的标题
plt.axis('off')  # 关闭坐标轴显示

# TODO 直方图匹配
result = histogram_matching(img,target)
# 显示直方图匹配结果图像
plt.subplot(233)  # 23:使输出的图像排版为2行3列，1:在2行3列中的第一行第三列显示
plt.imshow(result, cmap="gray")
plt.title("result")  # 设置图片的标题
plt.axis('off')  # 关闭坐标轴显示
# 计算直方图
hist_img = cv2.calcHist([img], [0], None, [255], [0, 255])
hist_target = cv2.calcHist([target], [0], None, [255], [0, 255])
hist_result = cv2.calcHist([result], [0], None, [255], [0, 255])
# 显示直方图
plt.subplot(234)
plt.plot(hist_img)
plt.title("hist_img")
plt.subplot(235)
plt.plot(hist_target)
plt.title("hist_target")
plt.subplot(236)
plt.plot(hist_result)
plt.title("hist_result")
plt.show()

# TODO 处理过程
# 首先进行中值滤波
medium_filter = MediumFilter(filter_size=3)
result_medium = medium_filter.convolution(img)
# 进行巴特沃斯低通滤波
result_lowpass = blpf(result_medium,100)
# # # 利用理想低通滤波器进行滤波
# result_lowpass, f_shift, h1, glpf = fourier_low_pass(result_medium,150)
# # 显示低通滤波器的K空间图像与掩模图像以及最终滤波后的K空间图像
# plt.subplot(131)
# plt.imshow(np.log(1 + abs(f_shift)))
# plt.title("img_f")
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(h1)
# plt.title("mask")
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(np.log(1 + abs(glpf)))
# plt.title("result_f")
# plt.axis('off')
# plt.show()
# 然后进行卷积核锐化，自己设计卷积核，设计为下列矩阵，这种卷积核满足归一化条件
kernel_sharpen = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]],dtype=np.float32)
# 卷积核锐化，由于图像矩阵类型为float类型，因此需要进行第二步灰阶范围转化
result_sharpen = cv2.filter2D(result_lowpass,-1,kernel_sharpen)
result_sharpen = cv2.convertScaleAbs(result_sharpen)
# 高斯核锐化
# gau = cv2.GaussianBlur(result_lowpass,(5,5),0.15)
# diff = cv2.addWeighted(result_lowpass[:,:],1,gau,-1,0)
# result_sharpen = cv2.addWeighted(result_lowpass[:,],1,diff,1,0)
# 画处理过程流程图
plt.subplot(221)
plt.imshow(img, cmap="gray")
plt.title("image")  # 设置图片的标题
plt.axis('off')  # 关闭坐标轴显示
plt.subplot(222)
plt.imshow(result_medium, cmap="gray")
plt.title("medium")  # 设置图片的标题
plt.axis('off')  # 关闭坐标轴显示
plt.subplot(223)
plt.imshow(result_lowpass, cmap="gray")
plt.title("lowpass")  # 设置图片的标题
plt.axis('off')  # 关闭坐标轴显示
plt.subplot(224)
plt.imshow(result_sharpen, cmap="gray")
plt.title("result")  # 设置图片的标题
plt.axis('off')  # 关闭坐标轴显示
plt.show()
# 画最终目标图像与处理结果对比图
plt.subplot(121)
plt.imshow(target, cmap="gray")
plt.title("target")  # 设置图片的标题
plt.axis('off')  # 关闭坐标轴显示
plt.subplot(122)
plt.imshow(result_sharpen, cmap="gray")
plt.title("result")  # 设置图片的标题
plt.axis('off')  # 关闭坐标轴显示
plt.show()
# 量化结果显示
print(f"原图与目标图像差异量化结果: {np.square((target - img)).sum()}")
print(f"中值滤波结果与目标图像差异量化结果: {np.square((target - result_medium)).sum()}")
print(f"低通滤波结果与目标图像差异量化结果: {np.square((target - result_lowpass)).sum()}")
print(f"锐化结果与目标图像差异量化结果: {np.square((target - result_sharpen)).sum()}")