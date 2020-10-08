import numpy as np
import cv2



def apply_img_filtr(img, filtr, new_image):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            new_image[y, x] = img[y, x]

    return new_image



if __name__ == '__main__':
    roberts_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    roberts_2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 0, 1]])
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    img1 = cv2.imread("zebra.jpg", 0)
    cv2.imshow("orig", img1)
    new_image = np.zeros(img1.shape, img1.dtype)
    print(kh)


    cv2.waitKey()
    cv2.destroyAllWindows()
