# this file is to test the vgg model 
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.preprocessing.image import ImageDataGenerator
import csv
import numpy as np
import math
import cv2
import tensorflow as tf
import glob as gb
import time
import os
import timeit

start = timeit.default_timer()



CATEGORIES = ["panel", "nopanel"]

# Input dirs

model_path = './final/split/'
path = model_path
model = tf.keras.models.load_model(os.path.join(path,'20191014-173338.hdf5'))

panel_panel = 0
panel_nopanel = 0
nopanel_panel = 0 
nopanel_nopanel = 0 
# test the panel result
panel_img_path = gb.glob("./location17/panel/*png")
nopanel_img_path = gb.glob(".//location17/nopanel/*png")


contour = {}
csvpath = './location17/vgg_predict.csv'
with open(csvpath, 'a') as csvfile:
    myFields = ['id','prediction','prediction_class','label']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()

num = 0
contour = {}
for path in panel_img_path:
    
    detected_path = path.split("/")[-1]
    contour['id'] = detected_path.split(".")[0]
    img = cv2.imread(path)
#     print(img.shape)
    IMG_SIZE = 150
    img1 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    testimg = (img1.reshape(-1, IMG_SIZE, IMG_SIZE, 3)).astype('int32')/255
    prediction_class = model.predict_classes(testimg)
    prediction  = model.predict(testimg)
    contour['prediction'] = prediction[0][0]
    contour['prediction_class'] = prediction_class[0][0]
    contour['label'] = 1
    if ((prediction_class[0][0]) == 1):
         panel_panel = panel_panel + 1  
    else:
        panel_nopanel = panel_nopanel + 1      
        
    with open(csvpath, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([contour['id'],contour['prediction'],contour['prediction_class'],contour['label']])
    csvfile.close()    

      
TP = panel_panel
FN = panel_nopanel
# test no panel result 
                        
for path in nopanel_img_path:
    detected_path = path.split("/")[-1]
    contour['id'] = detected_path.split(".")[0]
    img = cv2.imread(path)
    IMG_SIZE = 150
    img1 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    testimg = (img1.reshape(-1, IMG_SIZE, IMG_SIZE, 3)).astype('int32')/255
    prediction_class = model.predict_classes(testimg)
    prediction  = model.predict(testimg)
    contour['prediction'] = prediction[0][0]
    contour['prediction_class'] = prediction_class[0][0]
    contour['label'] = 0
    with open(csvpath, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([contour['id'],contour['prediction'],contour['prediction_class'],contour['label']])
    csvfile.close()  
   
    if ((prediction_class[0][0]) == 1):
         nopanel_panel = nopanel_panel + 1      
    else:
        nopanel_nopanel = nopanel_nopanel + 1
        
TN = nopanel_nopanel
FP =  nopanel_panel 

stop = timeit.default_timer()
time = {}
time['description'] = 'get vgg prediction on location17'
time['time'] = stop - start
csv_path =   './final/time.csv'
with open(csv_path, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([time['description'],time['time']])
csvfile.close() 
print('Time: ', stop - start)  
print(TP, FN,TN ,FP)



