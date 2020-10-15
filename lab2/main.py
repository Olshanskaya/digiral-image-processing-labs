import numpy as np
import cv2


def apply_img_filtr(img, filtr):
    new_image = np.zeros(img1.shape, dtype=np.int32)
    for y in range(new_image.shape[0]):
        for x in range(new_image.shape[1]):
            for i in range(3):
                for j in range(3):
                    new_image[y, x] += img[y + i - 1, x + j - 1] * filtr[i, j]
    # print(new_image)
    return new_image


def one_normalize_255(img):
    new_image = np.zeros(img.shape, dtype=np.uint8)
    max = np.max(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            new_image[y, x] = (abs(img[y, x])) * 255 / max
    return new_image


def normilize_255(img):
    max = np.max(img)
    min = np.min(img)
    print(img)
    new_image = np.zeros(img.shape, dtype=np.uint8)
    for y in range(new_image.shape[0]):
        for x in range(new_image.shape[1]):
            new_image[y, x] = (img[y, x] - min) * 255 / (max - min)
            # print(x, y, new_image[y, x])
    return new_image


def both_x_y(img_x, img_y):
    img_rez = np.zeros(img_x.shape, dtype=np.uint64)
    print(img_rez)
    for y in range(img_x.shape[0]):
        for x in range(img_x.shape[1]):
            img_rez[y, x] = np.sqrt(img_x[y, x] * img_x[y, x] + img_y[y, x] * img_y[y, x])
    print(img_rez)
    return img_rez


if __name__ == '__main__':
    roberts_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    roberts_2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    img1 = cv2.imread("lines.JPG", 0)
    # cv2.imshow("orig", img1)

    img_board = np.zeros((img1.shape[0] + 2, img1.shape[1] + 2), img1.dtype)
    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            img_board[y + 1, x + 1] = img1[y, x]
    cv2.imshow("orig", img_board)

    # roberts

    img_roberts_1 = apply_img_filtr(img_board, roberts_1)
    norm_rob_1 = one_normalize_255(img_roberts_1)
    #cv2.imshow("norm_rob_1", norm_rob_1)
    cv2.imwrite("norm_rob_x.jpg", norm_rob_1)
    img_roberts_2 = apply_img_filtr(img_board, roberts_2)
    norm_rob_2 = one_normalize_255(img_roberts_2)
    #cv2.imshow("norm_rob_2", norm_rob_2)
    cv2.imwrite("norm_rob_2.jpg", norm_rob_2)
    rezult_img = both_x_y(img_roberts_1, img_roberts_2)
    rezult_img = normilize_255(rezult_img)
    cv2.imshow("img_rob_y_x", rezult_img)
    cv2.imwrite("img_rob_y_x.jpg", rezult_img)


    # Sobel

    img_sobel_x = apply_img_filtr(img_board, sobel_x)
    norm_sob_1 = one_normalize_255(img_sobel_x)
    #cv2.imshow("norm_sob_x", norm_sob_1)
    img_sobel_y = apply_img_filtr(img_board, sobel_y)
    norm_sob_2 = one_normalize_255(img_sobel_y)
    #cv2.imshow("norm_sob_y", norm_sob_2)
    rezult_img = both_x_y(img_sobel_x, img_sobel_y)
    rezult_img = normilize_255(rezult_img)
    cv2.imshow("img_sob_y_x", rezult_img)

    # prewit

    img_prewit_x = apply_img_filtr(img_board, prewitt_x)
    norm_prew_x = one_normalize_255(img_prewit_x)
    #cv2.imshow("norm_prew_x", norm_prew_x)
    img_prewit_y = apply_img_filtr(img_board, prewitt_y)
    norm_prew_y = one_normalize_255(img_prewit_y)
    #cv2.imshow("norm_prew_y", norm_prew_y)
    rezult_img = both_x_y(img_prewit_x, img_prewit_y)
    rezult_img = normilize_255(rezult_img)
    cv2.imshow("img_prewit_y_x", rezult_img)


    cv2.waitKey()
    cv2.destroyAllWindows()
