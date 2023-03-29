import SimpleITK as sitk
import matplotlib.pyplot as plt
from Filters import BilateralFilter, MeanFilter, MediumFilter, GaussianFilter
import cv2
import numpy as np


def normalization(image):
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min())


# TODO 使用sitk load image
# 正确加载图像
image = sitk.ReadImage(r"brain.dcm")
array_image = np.squeeze(sitk.GetArrayFromImage(image))
## 展示读取出来的图像
plt.imshow(array_image, cmap="gray")
plt.title("source image") # 设置标题
plt.show()

## 保证array_image shape 为 (512, 512)
print(array_image.shape)
## 最大最小值归一化，将像素值归一化到 0-1 之间
array_image = normalization(array_image)


## 均值滤波 自己实现的方式与调用opencv函数 对比
# opencv 均值滤波
opencv_mean_filter = cv2.blur(array_image,(5,5))
# 用自己实现的均值滤波器 滤波
mean_filter = MeanFilter(filter_size=5)
my_mean_result = mean_filter.convolution(array_image)
# 结果显示对比
plt.subplot(1,2,1)
plt.imshow(opencv_mean_filter, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(my_mean_result, cmap="gray")
plt.show()
print(f"量化结果: {np.square((my_mean_result - opencv_mean_filter)).sum()}")

## TODO 中值滤波 自己实现的方式与调用opencv函数 对比
# opencv 中值滤波
opencv_medium_filter = cv2.medianBlur(array_image,5)
# 用自己实现的均值滤波器 滤波
medium_filter = MediumFilter(filter_size=5)
my_medium_result = medium_filter.convolution(array_image)
# 结果显示对比
plt.subplot(1,2,1)
plt.imshow(opencv_medium_filter, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(my_medium_result,cmap="gray")
plt.show()
print(f"量化结果: {np.square((my_medium_result- opencv_medium_filter)).sum()}")

## TODO 高斯滤波 自己实现的方式与调用opencv函数 对比
# opencv 高斯滤波
opencv_gaussian_filter = cv2.GaussianBlur(array_image, (3, 3), 1)
# 用自己实现的高斯滤波器 滤波
gaussian_filter = GaussianFilter(filter_size=3,sigma=1)
my_gaussian_result = gaussian_filter.convolution(array_image)
# 结果显示对比
plt.subplot(1,2,1)
plt.imshow(opencv_gaussian_filter, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(my_gaussian_result,cmap="gray")
plt.show()
print(f"量化结果: {np.square((my_gaussian_result- opencv_gaussian_filter)).sum()}")
## TODO 双边滤波 自己实现的方式与调用opencv函数 对比
# opencv 双边滤波
opencv_bilater_filter = cv2.bilateralFilter(array_image, d=3, sigmaColor=1, sigmaSpace=1)
# 用自己实现的双边滤波器 滤波
bilater_filter = BilateralFilter(filter_size=3,image=array_image,ssigma=1,gsigma=1)
my_bilater_result = bilater_filter.convolution(array_image)
#结果显示对比
plt.subplot(1,2,1)
plt.imshow(opencv_bilater_filter, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(my_bilater_result,cmap="gray")
plt.show()
print(f"量化结果: {np.square((my_bilater_result- opencv_bilater_filter)).sum()}")