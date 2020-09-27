from random import randint

import cv2
import numpy as np
import matplotlib.mlab as mlab
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
    histg = np.zeros((255, 1))

    for y in range(gray_img.shape[0]):
        for x in range(gray_img.shape[1]):
            i = gray_img[x, y]
            histg[i] = int(histg[i] + 1)

    histg_cv = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # cv2.imshow("gray_img", gray_img)

    # linear transforms
    linear_image = np.zeros(gray_img.shape, gray_img.dtype)
    all_cnt = gray_img.shape[0] * gray_img.shape[1]
    five_percent = int(all_cnt * 0.05)

    # print(max_hist, min_hist, non_zero)

    alpha = 0
    beta = 254
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
    equ_image_cv = cv2.equalizeHist(gray_img)

    trans_table = np.zeros((255, 1))
    sum_before = 0
    for y in range(255):
        trans_table[y] = sum_before + histg[y] * 255.0 / all_cnt
        sum_before += histg[y] * 255.0 / all_cnt

    equ_image = np.zeros(gray_img.shape, gray_img.dtype)
    for y in range(gray_img.shape[0]):
        for x in range(gray_img.shape[1]):
            equ_image[y, x] = np.clip(trans_table[gray_img[y, x]], 0, 255)

    histg_equalize = cv2.calcHist([equ_image], [0], None, [256], [0, 256])

    # visualization
    ideal = cv2.imread("Lenna_orig.png", 0)
    vis = np.concatenate((gray_img, equ_image), axis=1)
    vis = np.concatenate((vis, linear_image), axis=1)
    cv2.imshow("orig equalize linear", vis)

    '''
    plt.plot(histg, "b")
    plt.plot(histg_equalize, "r")
    plt.plot(histg_linear, "g")
    '''
    fig, ax = plt.subplots()
    x = np.arange(1, len(histg.transpose().tolist()[0]) + 1)
    ax.bar(x, histg.transpose().tolist()[0])
    fig, ax = plt.subplots()
    ax.bar(x, histg_equalize.transpose().tolist()[0])
    fig, ax = plt.subplots()
    ax.bar(x, histg_linear.transpose().tolist()[0])

    ax.set_facecolor('seashell')
    fig.set_facecolor('floralwhite')
    fig.set_figwidth(12)  # ширина Figure
    fig.set_figheight(6)  # высота Figure

    plt.show()

    '''
    plt.hist(gray_img)
    plt.hist(equ_image)
    plt.hist(gray_img)
    plt.show()
    '''

    cv2.waitKey()
    cv2.destroyAllWindows()

    '''
    vals = img.mean(axis=2).flatten()
    b, bins, patches = plt.hist(vals, 255)
    # plt.xlim([0, 100])
    plt.show()
    '''
