import numpy as np
import cv2


def apply_img_filtr(img, filtr, new_image):
    for y in range(new_image.shape[0]):
        for x in range(new_image.shape[1]):
            new_image[y, x] = img[y, x] * filtr[0, 0] + img[y + 1, x] * filtr[1, 0] + img[y + 2, x] * filtr[
                2, 0]
            new_image[y, x] += img[y, x + 1] * filtr[0, 1] + img[y + 1, x + 1] * filtr[1, 1] + img[
                y + 2, x + 1] * filtr[2, 1]
            new_image[y, x] += img[y, x + 2] * filtr[0, 2] + img[y + 1, x + 2] * filtr[1, 2] + img[
                y + 2, x + 2] * filtr[2, 2]
            # print(img[y, x], filtr[1, 1])
    print(new_image)
    return new_image


if __name__ == '__main__':
    roberts_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    roberts_2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 0, 1]])
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    img1 = cv2.imread("zebra.jpg", 0)
    # cv2.imshow("orig", img1)
    new_image = np.zeros(img1.shape, dtype=np.int32)
    img_roberts_1 = np.zeros(img1.shape, img1.dtype)
    img_roberts_2 = np.zeros(img1.shape, img1.dtype)
    img_roberts = np.zeros(img1.shape, img1.dtype)
    img_sobel_x = np.zeros(img1.shape, img1.dtype)
    img_sobel_y = np.zeros(img1.shape, img1.dtype)
    img_sobel = np.zeros(img1.shape, img1.dtype)
    img_prewit_y = np.zeros(img1.shape, img1.dtype)
    img_prewit_x = np.zeros(img1.shape, img1.dtype)
    img_prewit = np.zeros(img1.shape, img1.dtype)

    img_board = np.zeros((img1.shape[0] + 2, img1.shape[1] + 2), img1.dtype)
    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            img_board[y + 1, x + 1] = img1[y, x]
    cv2.imshow("orig", img_board)

    new_image = apply_img_filtr(img_board, sobel_x, new_image)
    sobelx64f = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    cv2.imshow("ne", sobelx64f)

    cv2.imshow("new_image", new_image.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
