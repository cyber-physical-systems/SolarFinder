# OpenCV lib
import os
import tensorflow as tf
import cv2
from skimage.segmentation import slic
from skimage import color
from skimage import data
from skimage import io
# Traverse files
import glob as gb
import tensorflow as tf
# Math lib
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import csv

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def kmeans(img):
    # K-means
    # Convert image to one dimension data
    img_ori = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    Z = img.reshape((-1, 3))
    # Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K =5
    # Run k-means
    # ret: compactness
    # labels:
    # centers: array of centers of clusters
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    res2_gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

    hist = res2_gray.ravel()
    hist = set(hist)
    hist = sorted(hist)
   # print(len(hist))
    threshold = []
    tag=[]
    tag1 = []
    tag_dilate3 = []
    tag_dilate5 = []
    tag_dilate7 = []
    tag_close3 = []
    tag_close5 = []
    tag_close7 = []
    for i in range(len(hist)-1):
        threshold.append(int(hist[i]/2 + hist[i+1]/ 2))
    #  no dilate , not accurate
    kernal3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernal5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernal7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for j in range(len(hist)-1):
        if j ==(len(hist)-2):
            dia=cv2.inRange(res2_gray, threshold[j], 255)
            tag.append(dia)
            tag_dilate3.append(cv2.dilate(dia, kernal3, iterations=1))
            tag_dilate5.append(cv2.dilate(dia, kernal5, iterations=1))
            tag_dilate7.append(cv2.dilate(dia, kernal7, iterations=1))
        else:
            dia =  cv2.inRange(res2_gray, threshold[j], threshold[j+1])
            tag.append(dia)
            tag_dilate3.append(cv2.dilate(dia, kernal3, iterations=1))
            tag_dilate5.append(cv2.dilate(dia, kernal5, iterations=1))
            tag_dilate7.append(cv2.dilate(dia, kernal7, iterations=1))

    for j in range(len(hist) - 1):
        if j == (len(hist) - 2):
            dia1 = cv2.inRange(res2_gray, threshold[j], 255)
            tag1.append(dia1)

            tag_close3.append(cv2.morphologyEx(dia1, cv2.MORPH_CLOSE, kernal3))
            tag_close5.append(cv2.morphologyEx(dia1, cv2.MORPH_CLOSE, kernal5))
            tag_close7.append(cv2.morphologyEx(dia1, cv2.MORPH_CLOSE, kernal7))
        else:
            dia1 = cv2.inRange(res2_gray, threshold[j], threshold[j + 1])
            tag1.append(dia1)
            tag_close3.append(cv2.morphologyEx(dia1, cv2.MORPH_CLOSE, kernal3))
            tag_close5.append(cv2.morphologyEx(dia1, cv2.MORPH_CLOSE, kernal5))
            tag_close7.append(cv2.morphologyEx(dia1, cv2.MORPH_CLOSE, kernal7))

    # return(tag,tag_dilate3,tag_close3, tag_dilate5,tag_close5, tag_dilate7, tag_close7 ,hist)
    return (tag, hist, tag_close3, tag_dilate5, tag_close5, tag_dilate7, tag_close7, hist)
# the kernel number is returned , use kernel 3 temporiarly.

# find contours based on kmeans method
def find_contours(img, mask_list):
    # Get the area of roof
    masks_length = len(mask_list)
    cont = []
    for i in range(0, masks_length):
        c, h = cv2.findContours(mask_list[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in c:
            cont.append(contour)
#     cv2.drawContours(img, cont, -1, (0, 0, 255), 2)
    return [img,cont]

# use size filter 
def filter_size(img,contour):
    image_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roof_area = cal_roofarea(image_grayscale)[0]
    cont = []
    for c in contour:
        area = cv2.contourArea(c)
        if (area >0):
            ratio = area / roof_area
            if ((area >800) & (ratio < 0.5)):
                cont.append(c)
    areas = []
    for i, co in enumerate(cont):
        areas.append((i, cv2.contourArea(co),co))

    a2 = sorted(areas, key=lambda d: d[1], reverse=True)
    # cv2.drawContours(img, cont, -1, (0, 0, 255), 2)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    return [img,a2]

# calculate the roof area so we can remove a part of the contours
def cal_roofarea(image):
    black = cv2.threshold(image, 0, 255, 0)[1]
    contours, hierarchy = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    area = [cv2.contourArea(c) for c in contours]
    roof_index = np.argmax(area)
    roof_cnt = contours[roof_index]
    # contourArea will return the wrong value if the contours are self-intersections
    roof_area = cv2.contourArea(roof_cnt)
    #print('roof area = '+ str(roof_area))
    return (roof_area,roof_cnt)

# calculate the mean pixel value in the contours
def getContourStat(img,contour):
  mask = np.zeros(img.shape,dtype="uint8")
  cv2.drawContours(mask, [contour], -1, 255, -1)
  mean,stddev = cv2.meanStdDev(img,mask=mask)
  return mean, stddev


# use to show the result of kmeans

def get_mask(img,mask_list):
    masks_length = len(mask_list)
    mask_color = [(255,0,0),(0,255,0),(0,0,255),(255,255,255),(128,128,128),(0,0,0)]
    for i in range(0, masks_length):
        img[mask_list[i]!= 0] = mask_color[i]
    return img


def pole(img, contour):
    ori_img = img.copy()
    image_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cont = cal_roofarea(image_grayscale)[1]
    cv2.drawContours(ori_img, cont, -1, (255, 0, 0), 3)
    #print(len(contour))
    contour_res =[]
    back = 1
    cnt = contour
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    pole = [leftmost,rightmost,topmost,bottommost]
    for point in pole:
        # check the distance with contours, biggest contour
        # when it is negative, means the point is outside the contours
        dist = cv2.pointPolygonTest(cont, point, True)
        # print(dist)
        if (dist <=0):
            back = 0
        else:
            pass

    return (ori_img,contour,back)
def rotate_rectangle(img_name,img, contour):

    shape= {}
    shape['id'] = img_name
# for c in contour:
    c = contour
    
    area = cv2.contourArea(c)
    x,y,w,h = cv2.boundingRect(c)
    ratiowh  =  min(float(w/h),float(h/w))
    shape['ratiowh'] = ratiowh

    ratioarea = float(area/(w*h))
    shape['ratioarea'] = ratioarea

    epsilon = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    approxlen = len(approx)
    shape['approxlen'] = approxlen


    #  the original num set to be -1 to be different no operation
    num_angle = 0
    num_angle90 = -1
    num_angle80 = -1
    num_angle70 = -1

    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)
    cv2.drawContours(img, [approx], -1, (255, 255, 255), 2)
    # mask = np.concatenate((mask, mask, mask), axis=-1)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contour_list = []
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # get the list of contours
    for points in contours[0]:
        x, y = points.ravel()
        contour_list.append([x, y])
    corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        #  decide whether the corner is on the contours
        if (cv2.pointPolygonTest(contours[0], (x, y), True) == 0):
            center_index = contour_list.index([x, y])
            length = len(contour_list)
            # get the point three before, and ignore the end point
            a_index = center_index - 5
            b_index = center_index + 5
            if ((a_index > 0) & (b_index > 0) & (a_index < length)& (b_index < length)):
                xa, ya = contour_list[a_index]
                xb, yb = contour_list[b_index]
                # print(x , y)
                # print(xa, ya)
                a = math.sqrt((x - xa) * (x - xa) + (y - ya) * (y - ya))
                b = math.sqrt((x - xb) * (x - xb) + (y - yb) * (y - yb))
                c = math.sqrt((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb))
                if ((a > 0) & (b > 0)):
                    if(((a * a + b * b - c * c) / (2 * a * b))<1) & (((a * a + b * b - c * c) / (2 * a * b) >-1)):
                        angle = math.degrees(math.acos((a * a + b * b - c * c) / (2 * a * b)))
                        num_angle =num_angle +1
                        # print(angle)
                        if (angle < 90):
                            num_angle90 = num_angle90 + 1
                        if (angle < 80):
                            num_angle80 = num_angle80 + 1
                        if (angle < 70):
                            num_angle70 = num_angle70 + 1
        cv2.circle(img, (x, y), 5, 255, -1)

    shape['numangle'] = num_angle
    shape['numangle90'] = num_angle90
    shape['numangle80'] = num_angle80
    shape['numangle70'] = num_angle70
#     print(shape)
    # with open(csv_path, 'a') as csv_file:
    #     writer = csv.writer(csv_file)
    #     # writer.writerow(['image_id','size','pole','mean','square'])
    #     writer.writerow([shape['id'],shape['ratiowh'], shape['ratioarea'],shape['approxlen'],shape['numangle'],shape['numangle90'],shape['numangle80'],shape['numangle70']])
    #     # for key, value in contour.items():
    #     #     writer.writerow([key, value])
    # csv_file.close()

    return(shape)
def mean(img,contour):
    cont_res = []
    ori_img= img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    image_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_filter = 0
    c = contour
    mean = getContourStat(image_grayscale,c)[0]
    hist = kmeans(img)[1]
    if (mean[0][0] <= (hist[2]+5)):
        # mean = 1 means panel
        mean_filter= 1

    else:
        # pass
        mean_filter = 0
    # print(mean)
#     cv2.drawContours(ori_img, cont_res, -1, (0, 0, 255), -1)
    return(ori_img,cont_res,mean_filter)

def main():

   
    path = './model/'
    model = tf.keras.models.load_model(os.path.join(path,'20191003-010747.hdf5'))
    num = 0 
    csvpath = './lrtrainhouse7poss.csv'
    with open(csvpath, 'a') as csvfile:
        myFields = ['id', 'image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','vgg','label']
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        writer.writeheader()
    csvfile.close()
    CATEGORIES = ["panel", "nopanel"]
    IMG_SIZE = 150
    img_path = gb.glob("./house7/*.png")
    # store the information of contours(the label) 
    for path in img_path:
        contour = {}
        img_name = path.split("/")[-1]
        img_name = img_name.split(".")[0]
        # print(img_name)
        # original image
        img = cv2.imread(path)
        # this is to show the contours so we can label right
        img_contour = img.copy()
#         tag = kmeans(img.copy())[2]
        tag = kmeans(img)[2]
#         masks = get_mask(img, tag)
        # get the contours
        img_contours= find_contours(img, tag)[0]
        contours = find_contours(img, tag)[1]
        # filter: to remove the contours which is less than 1 block of solar panel
        img_size = filter_size(img, contours)[0]
        contourinfo = filter_size(img, contours)[1]
        # conotur_num is to tag the contours on the image
        contour_num = 0
        rank = 0
        for i, area, c in contourinfo:
            contour = {}
            rank = rank + 1
            contour['id'] = str(img_name) + '_' + str(rank)
            print(contour['id'])
            contour['image'] =  str(img_name)
            contour['size'] = area
#             contour['cont'] = c
            contour['pole'] = pole(img.copy(), c)[2]
            # print(contour['pole'])
            # if the value is 1, means it maybe panel
            contour['mean'] = mean(img.copy(), c)[2]
            # print(contour['mean'])
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            sq = 4 * math.pi * area / (perimeter * perimeter)
            contour['square'] = sq
            # print(sq)
            shape = rotate_rectangle(img_name,img.copy(), c)
            contour['ratiowh'] =  shape['ratiowh']
            contour['ratioarea'] = shape['ratioarea']
            contour['approxlen'] = shape['approxlen']
            contour['numangle'] = shape['numangle']
            contour['numangle90'] = shape['numangle90']
            contour['numangle70'] = shape['numangle70']
            csv_path = './contourlabel1.csv'
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if(row['id']==contour['id']):
#                         print(row['id'],row['label'])
                        contour['label'] = row['label']
#                         num = num + 1
                        vgg_image = img.copy()
                        mask = np.zeros_like(img)
                        img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        mask = cv2.drawContours(img2gray, [c], 0, (255, 255, 255, 1), -1)
                        img_result = cv2.bitwise_or(vgg_image, vgg_image, mask=mask)
                        cv2.imshow(img_result)
                        cv2.waitKey(0)
                        img_result = cv2.resize(img_result, (IMG_SIZE, IMG_SIZE))
                        testimg = (img_result.reshape(-1, IMG_SIZE, IMG_SIZE, 3)).astype('int32')/255
                        prediction = model.predict(testimg)
                        contour['vgg'] =  prediction[0][0] 
            #             if ((prediction[0][0]) > (0.5)):
            #                 contour['vgg'] = 1
            #             else:
            #                 contour['vgg'] = 0 
                        print(contour)
                        with open(csvpath, 'a') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([contour['id'], contour['image'],contour['size'],contour['pole'],contour['mean'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['vgg'],contour['label']])
                        csvfile.close()
    print('finish')
main()






