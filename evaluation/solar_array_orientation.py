import os
import cv2
from skimage.segmentation import slic
from skimage import color
from skimage import data
from skimage import io
# Traverse files
import glob as gb
# Math lib
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import csv
import os.path as path

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def cal_roofarea(image):
    black = cv2.threshold(image, 0, 255, 0)[1]
    # cv2.imshow('img', black)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    area = [cv2.contourArea(c) for c in contours]
    roof_index = np.argmax(area)
    roof_cnt = contours[roof_index]
    # contourArea will return the wrong value if the contours are self-intersections
    roof_area = cv2.contourArea(roof_cnt)
    #print('roof area = '+ str(roof_area))
    return (roof_area,roof_cnt)




img_path = './panel/'
contours_path = './projects/data/panel/'
csv_path = './vggsvmlogicalregression2features.csv'
with open('./data/orientation_positive.csv', 'a') as csvfile:
    myFields = ['id', 'image','contour','roof']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()
# num_all = 0
# num_5 =0
# num_10 =0
# num_15 = 0
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        orientation = {}
        if(row['label'] == '1' and row['vggsvmlogicalregression2features']=='1'):
            orientation['id']  =row['id']
            orientation['image'] = row['image']
            img_name = row['image']
            contour_name = row['id']
            image_path = img_path + img_name + '.png'
            contour_path = img_path + contour_name + '.png'
            if path.exists(image_path):
                if path.exists(contour_path ):
                    img_roof = cv2.imread(image_path)
                    img_contour = cv2.imread(contour_path)
                    # cal_roofarea(img)
                    img_contour_grayscale = cv2.cvtColor(img_contour, cv2.COLOR_BGR2GRAY)
                    cont_contour = cal_roofarea(img_contour_grayscale)[1]
                    cv2.drawContours(img_contour, cont_contour, -1, (0, 0, 255), -1)
                    rect_contour = cv2.minAreaRect(cont_contour)
                    orientation['contour'] = rect_contour[2]
                    # print(rect_contour[2])
                    # box_contour = cv2.boxPoints(rect_contour)
                    # box = np.int0(box)
                    # print(box)
                    # cv2.drawContours(img_contour, [box], 0, (255, 0, 0), 1)
                    img_roof_grayscale = cv2.cvtColor(img_roof, cv2.COLOR_BGR2GRAY)
                    cont_roof = cal_roofarea(img_roof_grayscale )[1]
                    # cv2.drawContours(img , cont, -1, (0, 0, 255), -1)
                    rect_roof = cv2.minAreaRect(cont_roof)
                    orientation['roof'] = rect_roof[2]
                    # print(rect[2])
                    # box = cv2.boxPoints(rect)
                    # box = np.int0(box)
                    # # print(box)
                    # cv2.drawContours(img, [box], 0, (255, 0, 0), 1)
                    #
                    # x, y, w, h = cv2.boundingRect(cont)
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # print(x,y,w,h)
                    # print(cal_roofarea(cont)[0])
                    print(orientation)
                    # cv2.imshow('img', img_contour)
                    # cv2.waitKey(0)
                    with open('./data/orientation_positive.csv', 'a') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([orientation['id'], orientation['image'], orientation['contour'],orientation['roof']])
                    csvfile.close()


csvfile.close()