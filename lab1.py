import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("Lenna.png")
    cv2.imshow("img", img)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert isinstance(gray_img, object)
    cv2.imshow("gray_img", gray_img)
    
    vals = img.mean(axis=2).flatten()
    b, bins, patches = plt.hist(vals, 255)
    #plt.xlim([0, 100])
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()
