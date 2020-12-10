import numpy as np
import cv2

default_structure = np.ones((3, 3))


def indx_mod(index):
    if index < 0:
        return 0
    else:
        return index


def my_dilation(img_bin, kernel):
    kernel_x, kernel_y = kernel.shape
    rezult = np.zeros((img_bin.shape[0], img_bin.shape[1]))
    img_x, img_y = img_bin.shape

    half_x = (kernel_x - 1) // 2
    half_y = (kernel_y - 1) // 2

    # print(kernel_x, (int(np.ceil((kernel_x - 1) // 2.0))), ((kernel_x - 1) // 2))
    print(len(img_bin), img_bin.shape[0], img_bin.shape[1])

    for i in range(img_x):
        for j in range(img_y):

            # находим ту часть А, на которую будем накладывать B
            a_left = indx_mod(i - half_x)
            a_right = i + (kernel_x - half_x)
            a_up = indx_mod(j - half_y)
            a_down = j + (kernel_y - half_y)
            a_part = img_bin[a_left:a_right, a_up:a_down]

            # точно ли B помещается
            b_left = int(np.fabs(i - half_x)) if i - half_x < 0 else 0
            b_up = int(np.fabs(j - half_y)) if j - half_y < 0 else 0
            b_right = kernel_x - 1 - (i + (kernel_x - half_x) - img_bin.shape[0]) if i + (
                    kernel_x - half_x) > img_bin.shape[0] else kernel_x - 1
            b_down = kernel_y - 1 - (j + (kernel_y - half_y) - img_bin.shape[1]) if j + (
                    kernel_y - half_y) > img_bin.shape[1] else kernel_y - 1
            # накладываемый B
            b = kernel[b_left:b_right + 1, b_up:b_down + 1]

            # B является подмножетсвом A
            if np.array_equal(np.logical_and(a_part, b), b):
                rezult[i, j] = 1

            # объединение A c B не явля пустым множетсвом
            if np.logical_and(b, a_part).any():
                rezult[i, j] = 1

    return rezult


def my_erosion(img_bin, kernel):
    kernel_x, kernel_y = kernel.shape
    rezult = np.zeros((img_bin.shape[0], img_bin.shape[1]))
    img_x, img_y = img_bin.shape

    half_x = (kernel_x - 1) // 2
    half_y = (kernel_y - 1) // 2

    # print(kernel_x, (int(np.ceil((kernel_x - 1) // 2.0))), ((kernel_x - 1) // 2))
    print(len(img_bin), img_bin.shape[0], img_bin.shape[1])

    for i in range(img_x):
        for j in range(img_y):

            # находим ту часть А, на которую будем накладывать B
            a_left = indx_mod(i - half_x)
            a_right = i + (kernel_x - half_x)
            a_up = indx_mod(j - half_y)
            a_down = j + (kernel_y - half_y)
            a_part = img_bin[a_left:a_right, a_up:a_down]

            # точно ли B помещается
            b_left = int(np.fabs(i - half_x)) if i - half_x < 0 else 0
            b_up = int(np.fabs(j - half_y)) if j - half_y < 0 else 0
            b_right = kernel_x - 1 - (i + (kernel_x - half_x) - img_bin.shape[0]) if i + (
                    kernel_x - half_x) > img_bin.shape[0] else kernel_x - 1
            b_down = kernel_y - 1 - (j + (kernel_y - half_y) - img_bin.shape[1]) if j + (
                    kernel_y - half_y) > img_bin.shape[1] else kernel_y - 1
            # накладываемый B
            b = kernel[b_left:b_right + 1, b_up:b_down + 1]

            # B является подмножетсвом A
            if np.array_equal(np.logical_and(a_part, b), b):
                rezult[i, j] = 1
    return rezult


if __name__ == '__main__':
    img_orig = cv2.imread("zebra.jpg", 0)
    # cv2.imshow("orig", img_orig)

    # бинаризация Оцу
    ret2, th2 = cv2.threshold(img_orig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Otsu", th2)

    # ядро
    k = 4
    kernel = np.ones((k, k), np.uint8)

    '''
    # проверка erosion
    opencv_erosion = cv2.erode(th2, kernel, iterations=1)
    cv2.imshow("opencv_erosion", opencv_erosion)
    my_eros = my_erosion(th2, kernel)
    cv2.imshow("my_erosion", my_eros)
    '''

    '''
    # проверка dilation
    opencv_dilate = cv2.dilate(th2, kernel, iterations=1)
    cv2.imshow("opencv_dilate", opencv_dilate)
    my_dilation = my_dilate(th2, kernel)
    cv2.imshow("my_dilation", my_dilation)
    '''

    # opencv открытие
    erosion_op = cv2.erode(th2, kernel, iterations=1)
    opencv_opening = cv2.dilate(erosion_op, kernel, iterations=1)
    cv2.imshow("opencv_opening", opencv_opening)

    # ручное открытие
    my_erosion_op = my_erosion(th2, kernel)
    my_opening = my_dilation(my_erosion_op, kernel)
    cv2.imshow("my_opening", my_opening)

    # opencv закрытие
    dilate_cl = cv2.dilate(th2, kernel, iterations=1)
    opencv_closening = cv2.erode(dilate_cl, kernel, iterations=1)
    cv2.imshow("opencv_closening", opencv_closening)

    my_dilate_cl = my_dilation(th2, kernel)
    my_closening = my_erosion(my_dilate_cl, kernel)
    cv2.imshow("my_closening", my_closening)

    cv2.waitKey()
    cv2.destroyAllWindows()
