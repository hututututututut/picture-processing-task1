import numpy as np
import cv2


class AbstractFilter:

    def __init__(self, filter_size=3):
        assert filter_size % 2 == 1, 'Only odd kernel size is supported for the ease of implementation!'
        self.filter_size = filter_size

    def kernel_maker(self):
        raise NotImplementedError

    def set_kernel_size(self, filter_size):
        self.filter_size = filter_size

    def convolution(self, image):
        filter = self.kernel_maker()

        pad_num = int((self.filter_size - 1) / 2)
        pad_image = np.pad(image, (pad_num, pad_num), mode="constant", constant_values=0)

        m, n = pad_image.shape
        output_image = np.zeros_like(pad_image)

        for i in range(pad_num, m - pad_num):
            for j in range(pad_num, n - pad_num):
                output_image[i, j] = np.sum(
                    filter * pad_image[(i - pad_num): (i + pad_num + 1), (j - pad_num): (j + pad_num + 1)]) 

        output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]
        return output_image


class MeanFilter(AbstractFilter):

    def __init__(self, filter_size):
        super(MeanFilter, self).__init__(filter_size)
        self.kernel_array = np.ones((self.filter_size, self.filter_size))

    def kernel_maker(self):
        return self.kernel_array / np.sum(self.kernel_array)


class GaussianFilter(AbstractFilter):

    def __init__(self, filter_size, sigma):
        super(GaussianFilter, self).__init__(filter_size)
        self.r = int((filter_size - 1) / 2)
        self.sigma = sigma
        self.kernel_array = np.zeros((self.filter_size, self.filter_size))

    def kernel_maker(self):        
        ## TODO 构建高斯滤波器的 kernel
        for i in range(-self.r, self.r + 1):
            for j in range(-self.r, self.r + 1):
                # 因为高斯核前面的系数相同，在此只需写出指数部分，最终实现归一化即可
                self.kernel_array[i + self.r][j + self.r] = np.exp(-float(float((i * i + j * j)) / 2 * self.sigma * self.sigma))
        return self.kernel_array / np.sum(self.kernel_array)

class BilateralFilter(AbstractFilter):

    def __init__(self, filter_size, image, ssigma, gsigma):
        super(BilateralFilter, self).__init__(filter_size)
        self.r = int((filter_size - 1) / 2)
        self.image = image
        self.row, self.col = image.shape

        self.ssigma = ssigma
        self.gsigma = gsigma
        self.kernel_gauss = np.zeros((self.filter_size, self.filter_size))
        self.kernel_space = np.zeros((self.filter_size, self.filter_size))
        self.kernel_array = np.zeros((self.filter_size, self.filter_size))
    def kernel_maker(self,i,j):
        ## TODO 实现双边滤波
        for a in range(-self.r, self.r + 1):
            for b in range(-self.r, self.r + 1):
                # 因为高斯核前面的系数相同，在此只需写出指数部分，最终实现归一化即可
                self.kernel_gauss[a + self.r][b + self.r] = np.exp(-float(float((a * a + b * b)) / 2 * self.gsigma * self.gsigma))
        pad_image = np.pad(self.image, (self.r, self.r), mode="constant", constant_values=0)
        for m in range(-self.r, self.r + 1):
            for n in range(-self.r, self.r + 1):
                self.kernel_space[m + self.r][n + self.r] = np.exp(-pow((pad_image[i][j] - pad_image[i + m][j + n]), 2) / 2 * self.ssigma * self.ssigma)
                self.kernel_array[m + self.r][n +self.r] = self.kernel_space[m + self.r][n + self.r] * self.kernel_gauss[m + self.r][n + self.r]
        return self.kernel_array / np.sum(self.kernel_array)

    def convolution(self, image):
        pad_num = int((self.filter_size - 1) / 2)
        pad_image = np.pad(self.image, (pad_num, pad_num), mode="constant", constant_values=0)
        m, n = pad_image.shape
        output_image = np.zeros_like(pad_image)
        for i in range(pad_num, m - pad_num):
            for j in range(pad_num, n - pad_num):
                filter = self.kernel_maker(i,j)
                output_image[i, j] = np.sum(filter * pad_image[(i - pad_num): (i + pad_num + 1), (j - pad_num): (j + pad_num + 1)])
        output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]
        return output_image



class MediumFilter(AbstractFilter):

    def __init__(self, filter_size):
        super(MediumFilter, self).__init__(filter_size)
        assert filter_size % 2 == 1, 'Only odd kernel size is supported for the ease of implementation!'
        self.filter_size = filter_size

    def convolution(self, image):
        ## TODO 实现中值滤波
        pad_num = int((self.filter_size - 1) / 2)
        pad_image = np.pad(image, (pad_num, pad_num), mode="constant", constant_values=0)
        m, n = pad_image.shape
        output_image = np.zeros_like(pad_image)
        for i in range(pad_num, m - pad_num):
            for j in range(pad_num, n - pad_num):
                output_image[i, j] = np.median(pad_image[(i - pad_num): (i + pad_num + 1), (j - pad_num): (j + pad_num + 1)])
        output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]
        return output_image

        
