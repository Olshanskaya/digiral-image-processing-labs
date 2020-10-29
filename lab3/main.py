import numpy as np
import cv2
import random


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
            #print( int(img2[y, x]), img2[y, x], abs(img1[y, x] - img2[y, x]), ans)
    ans /= img1.shape[0] * img1.shape[1]
    return ans


if __name__ == '__main__':
    kernel_size = 9

    img_orig = cv2.imread("zebra.jpg", 0)
    img_noise = add_noise(img_orig)

    cv2.imshow("img_noise", img_noise)
    cv2.imshow("img_orig", img_orig)

    print(average_modulus_of_difference(img_orig, img_orig))
    print(average_modulus_of_difference(img_orig, img_noise))

    # gauss

    sigma = 0.5
    ker_line_x = cv2.getGaussianKernel(kernel_size, sigma)
    # print(ker_line_x)
    kernel = np.multiply(ker_line_x.T, ker_line_x)
    # kernel = kernel * 255  / kernel.max()
    cv2.imwrite("kernelGaussian.jpg", kernel)
    # output_gauss = img_noise.copy()

    output_gauss = cv2.filter2D(img_noise, -1, kernel)
    cv2.imshow("output_gauss", output_gauss)

    gaus = cv2.GaussianBlur(img_noise, (kernel_size, kernel_size), sigma)
    cv2.imshow("Gaussian blur opencv", gaus)

    print(average_modulus_of_difference(img_orig, output_gauss))

    # cv2.imshow("output", output)

    # b = cv2.sepFilter2D(img_noise, -1, kernel, kernel)
    # cv2.imshow('a', b)

    # for y in range(kernel_size):
    #    for x in range(kernel_size):

    # Bilateral

    # Non-local Means

    # img_board = np.zeros((img1.shape[0] + 2, img1.shape[1] + 2), img1.dtype)

    # cv2.imshow("img_prewit_y_x", rezult_img_prewit)

    # all
    # all_sravnenie = np.concatenate((rezult_img_roberts, rezult_img_Sobel), axis=1)

    # cv2.imwrite("all_sravnenie.jpg", all_sravnenie)

    cv2.waitKey()
    cv2.destroyAllWindows()
