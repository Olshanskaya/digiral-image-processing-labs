import numpy as np
import cv2
import random


def normilize_255(img123):
    max = np.max(img123)
    min = np.min(img123)
    print(img123)
    new_image = np.zeros(img123.shape, dtype=np.uint8)
    if max == min:
        return new_image
    for y in range(img123.shape[0]):
        for x in range(img123.shape[1]):
            new_image[y, x] = (img123[y, x] - min) * 255 / (max - min)
    return new_image


def apply_img_filtr_sobel(img, filtr, kernel_size):
    half_kernel_size = int(kernel_size / 2)
    a = int(img.shape[0] + 2 * half_kernel_size)
    b = int(img.shape[1] + 2 * half_kernel_size)
    orig_img = np.zeros((a, b), dtype=np.int32)
    new_image = np.zeros((a, b), dtype=np.int32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            orig_img[y + half_kernel_size, x + half_kernel_size] = img[y, x]

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for i in range(kernel_size):
                for j in range(kernel_size):
                    a = orig_img[y + i - half_kernel_size, x + j - half_kernel_size]
                    b = filtr[i, j]
                    new_image[y + half_kernel_size, x + half_kernel_size] += a * b

    return_img = normilize_255(new_image)
    return return_img


def add_noise(img):
    new_img = img.copy()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            val = new_img[y, x] + random.gauss(0, 10)
            if val < 0:
                new_img[y, x] = 0
            elif val > 255:
                new_img[y, x] = 255
            else:
                new_img[y, x] = val
    return new_img


def average_modulus_of_difference(img1, img2):
    ans = 0
    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            ans += abs(int(img2[y, x]) - int(img1[y, x]))
    ans /= img1.shape[0] * img1.shape[1]
    return ans


if __name__ == '__main__':
    kernel_size = 5

    img_orig = cv2.imread("zebra.jpg", 0)
    img_noise = add_noise(img_orig)

    cv2.imshow("img_noise", img_noise)
    cv2.imshow("img_orig", img_orig)

    print(average_modulus_of_difference(img_orig, img_orig))
    print(average_modulus_of_difference(img_orig, img_noise))

    # gauss

    sigma = 0.65
    ker_line_x = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.multiply(ker_line_x.T, ker_line_x)
    # kernel = kernel * 255  / kernel.max()
    cv2.imwrite("kernelGaussian.jpg", kernel)


    output_gauss = cv2.filter2D(img_noise, -1, kernel)
    cv2.imshow("output_gauss", output_gauss)

    # output_my_filter2D = apply_img_filtr_sobel(img_noise, kernel, kernel_size)
    # cv2.imshow("output_my_filter2D", output_my_filter2D)

    # gaus = cv2.GaussianBlur(img_noise, (kernel_size, kernel_size), sigma)
    # cv2.imshow("Gaussian blur opencv", gaus)

    print(average_modulus_of_difference(img_orig, output_gauss))

    # cv2.imshow("output", output)

    # b = cv2.sepFilter2D(img_noise, -1, kernel, kernel)
    # cv2.imshow('a', b)


    cv2.waitKey()
    cv2.destroyAllWindows()
