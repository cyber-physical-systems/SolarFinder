#openCV lib
import os


import cv2
import glob as gb
import numpy as np
import csv
import math

def getContourStat(img, contour):
    mask = np.zeros((800,800), dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean, stddev = cv2.meanStdDev(img, mask=mask)
    return mean, stddev

def main():
    # the file store the contour file
    csvpath_all = '/aul/homes/1019/split/feature_all.csv'
    with open(csvpath_all, 'a') as csvfile:
        myFields = ['id','location','image', 'size','pole','mean','stddev','b_mean','g_mean','r_mean','b_stddev','g_stddev','r_stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label','vgg_pro','vgg_class']
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        writer.writeheader()
    csvfile.close()
    # image path
    
    img_path_panel = gb.glob('/aul/homes/final_contour/house'+ str(i) +'/panel/*.png')
    img_path_nopanel = gb.glob('/aul/homes/final_contour/house'+ str(i) +'/nopanel/*.png')
    npy_path = '/aul/homes/dataset/dataset930/house'+ str(i) +'/contour/'
    csv_path = '/aul/homes/1019/split/feature17.csv'
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            contour = row
            i = contour['location'][-1]
            if (i =='0'):
                i = '10'
            if (contour['label']=='1'):
                path = img_path_panel
            if (contour['label']=='0'):
                path = img_path_nopanel
            img = cv2.imread(path)
            c = np.load(npy_path + contour['image'] + '.npy')
            image_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean = getContourStat(image_grayscale, c)[0]
            stddev =getContourStat(image_grayscale, c)[1]
            contour['mean'] = mean[0][0]
            contour['stddev'] = stddev[0][0]
            mean_all = getContourStat(img, c)[0]
            stddev_all = getContourStat(img, c)[1]
            contour['b_mean'] =  mean_all[0][0]
            contour['g_mean'] =  mean_all[1][0]
            contour['r_mean'] =  mean_all[2][0]
            contour['b_stddev'] =  stddev_all[0][0]
            contour['g_stddev'] =  stddev_all[1][0]
            contour['r_stddev'] =  stddev_all[2][0]   

            with open(csvpath_all, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([contour['id'], contour['location'],contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['b_mean'],contour['g_mean'],contour['r_mean'],contour['b_stddev'],contour['g_stddev'],contour['r_stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label'],contour['vgg_pro'],contour['vgg_class']])
        csvfile.close()
        
main()











