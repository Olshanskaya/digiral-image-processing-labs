import cv2
import numpy as np
import matplotlib.pyplot as plt


def round_fun(img2, a, b):
    rez = (img2[y, x] - a) * 255 / (b - a)
    if rez < 0:
        return 0
    if rez > 254:
        return 254
    return rez


if __name__ == '__main__':
    img = cv2.imread("Lenna.png")
    # cv2.imshow("orig", img)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert isinstance(gray_img, object)
    histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    # cv2.imshow("gray_img", gray_img)

    # linear transforms
    linear_image = np.zeros(gray_img.shape, gray_img.dtype)
    five_percent = int(gray_img.shape[0] * gray_img.shape[1] * 0.05)
    max_hist = histg.max()
    min_hist = histg.min()
    non_zero = np.count_nonzero(histg, axis=None)
    # print(max_hist, min_hist, non_zero)
    left = 0
    while histg[left] == 0:
        left = left + 1
    right = 254
    while histg[right] == 0:
        right = right - 1
    alpha = left
    beta = right
    # print(left, right)
    while five_percent > 0:
        if histg[alpha] < histg[beta]:
            five_percent = five_percent - histg[alpha]
            alpha = alpha + 1
        else:
            five_percent = five_percent - histg[beta]
            beta = beta - 1
    # print(alpha, beta)
    # print(gray_img.shape)

    for y in range(gray_img.shape[0]):
        for x in range(gray_img.shape[1]):
            linear_image[y, x] = np.clip(round_fun(gray_img, alpha, beta), 0, 255)
    histg_linear = cv2.calcHist([linear_image], [0], None, [256], [0, 256])

    # equalization
    equ_image = cv2.equalizeHist(gray_img)
    histg_equalize = cv2.calcHist([equ_image], [0], None, [256], [0, 256])

    # visualization
    vis = np.concatenate((gray_img, equ_image), axis=1)
    vis = np.concatenate((vis, linear_image), axis=1)
    cv2.imshow("orig equalize linear", vis)

    # plt.plot(histg, "b", label='line 1')
    plt.plot(histg, "b")
    plt.plot(histg_equalize, "r")
    plt.plot(histg_linear, "g")
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()

    '''
    vals = img.mean(axis=2).flatten()
    b, bins, patches = plt.hist(vals, 255)
    # plt.xlim([0, 100])
    plt.show()
    '''
