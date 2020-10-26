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

if __name__ == '__main__':
    roberts_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

    img1 = cv2.imread("zebra.jpg", 0)

    img_board = np.zeros((img1.shape[0] + 2, img1.shape[1] + 2), img1.dtype)



    #cv2.imshow("img_prewit_y_x", rezult_img_prewit)



    # all
    #all_sravnenie = np.concatenate((rezult_img_roberts, rezult_img_Sobel), axis=1)

    #cv2.imwrite("all_sravnenie.jpg", all_sravnenie)

    cv2.waitKey()
    cv2.destroyAllWindows()
