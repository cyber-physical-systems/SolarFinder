# OpenCV lib
import os


import cv2
import glob as gb
import numpy as np
import csv
import math

def getContourStat(img, contour):
    mask = np.zeros(img.shape, dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean, stddev = cv2.meanStdDev(img, mask=mask)
    return mean, stddev

def cal_roofarea(image):
    black = cv2.threshold(image, 0, 255, 0)[1]
    _,contours, hierarchy = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    area = [cv2.contourArea(c) for c in contours]
    roof_index = np.argmax(area)
    roof_cnt = contours[roof_index]
    # contourArea will return the wrong value if the contours are self-intersections
    roof_area = cv2.contourArea(roof_cnt)
    #print('roof area = '+ str(roof_area))
    return (roof_area,roof_cnt)

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
    _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

    return(shape)

def main():
    # the file store the contour file
    csvpath_all = '/aul/homes/final_contour/house3/contour_all.csv'
    with open(csvpath_all, 'a') as csvfile:
        myFields = ['id', 'image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        writer.writeheader()
    csvfile.close()
    
    csvpath_yes = '/aul/homes/final_contour/house3/contour_features.csv'
    with open(csvpath_yes, 'a') as csvfile:
        myFields = ['id', 'image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        writer.writeheader()
    csvfile.close()
    
    csvpath_no = '/aul/homes/final_contour/house3/no_contour_features.csv'
    with open(csvpath_no, 'a') as csvfile:
        myFields = ['id', 'image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        writer.writeheader()
    csvfile.close()
    
    
    img_path = gb.glob('/aul/homes/final_contour/house3/panel/*.png')
    npy_path = '/aul/homes/dataset/dataset930/house3/contour/'
    for path in img_path:
        contour = {}
        contour_name = path.split("/")[-1]
        contour_name = contour_name.split(".")[0]
        contour['id'] = contour_name
        img_name = contour_name.split("_")[0]
#         print(img_name)
        c = np.load(npy_path + contour_name + '.npy')
#         print(c)
        #  the file store images
        img = cv2.imread('/aul/homes/dataset/dataset930/house3/roof/'+ img_name + '.png')
        cv2.drawContours(img, c, -1, (0, 255, 0), 3)
        image_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = getContourStat(image_grayscale, c)[0]
        stddev =getContourStat(image_grayscale, c)[1]
        contour['mean'] = mean[0][0]
        contour['stddev'] = stddev[0][0]
         
        contour['image'] =  str(img_name)
        contour['size'] = cv2.contourArea(c)
#             contour['cont'] = c
        contour['pole'] = pole(img.copy(), c)[2]
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
        contour['label'] = str(1)
        #  the file to store the mean value and stddev 
        with open(csvpath_all, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([contour['id'], contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
        csvfile.close()
        with open(csvpath_yes, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([contour['id'], contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
        csvfile.close()
    print('finish')
    
    
    img_path = gb.glob('/aul/homes/final_contour/house3/nopanel/*.png')
    npy_path = '/aul/homes/dataset/dataset930/house3/contour/'
    for path in img_path:
        contour = {}
        contour_name = path.split("/")[-1]
        contour_name = contour_name.split(".")[0]
        contour['id'] = contour_name
        img_name = contour_name.split("_")[0]
#         print(img_name)
        c = np.load(npy_path + contour_name + '.npy')
#         print(c)
        #  the file store images
        img = cv2.imread('/aul/homes/dataset/dataset930/house3/roof/'+ img_name + '.png')
        cv2.drawContours(img, c, -1, (0, 255, 0), 3)
        image_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = getContourStat(image_grayscale, c)[0]
        stddev =getContourStat(image_grayscale, c)[1]
        contour['mean'] = mean[0][0]
        contour['stddev'] = stddev[0][0]
         
        contour['image'] =  str(img_name)
        contour['size'] = cv2.contourArea(c)
#             contour['cont'] = c
        contour['pole'] = pole(img.copy(), c)[2]
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
        contour['label'] = str(0)
        #  the file to store the mean value and stddev 
        with open(csvpath_all, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([contour['id'], contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
        csvfile.close()
        with open(csvpath_no, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([contour['id'], contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
        csvfile.close()
    print('finish')


    

main()








