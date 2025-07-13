import cv2
import glob
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

count = 0

for file in glob.glob("D:/CT_scans/normal_image_dir/*.jpg"):
    im_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    print(im_gray.shape)

    im_gray = cv2.resize(im_gray, (512, 512))

    print("After reshape: "+str(im_gray.shape))

    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

    plt.imshow(im_bw)
    plt.show()

    cv2.imwrite('resized/{}.jpg'.format(count), im_bw)
    count += 1
