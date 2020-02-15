import os
import cv2
import glob as gb
num = 0
img_path = gb.glob("./*.png")
for path in img_path:
    img_name = path.split("/")[-1]
    img = cv2.imread(path)
    if ((num % 5) == 0):
        cv2.imwrite(os.path.join('./' + img_name),img)
    num = num + 1
        