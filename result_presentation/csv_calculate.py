import pandas
import pandas as pd
import pickle
import numpy as np
import csv 
image_panel = []
image_nopanel = []
vgg_panel_panel = []
vgg_panel_nopanel = []
vgg_nopanel_panel = []
vgg_nopanel_nopanel = []

lr_panel_panel = []
lr_panel_nopanel = []
lr_nopanel_panel = []
lr_nopanel_nopanel = []
csv_path = './nosplit/test/vgg_predict-Copy2.csv'
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if ((row['label'] == '1') and (row['image'] not in image_panel)):
            image_panel.append(row['image'])
        if ((row['label'] == '0') and (row['image'] not in image_nopanel)):
            image_nopanel.append(row['image'])
            
csvfile.close()
print(len(image_panel),len(image_nopanel))

with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if ((row['prediction_class'] == '1') and (row['image'] not in vgg_panel_panel) and (row['label'] == '1')):
            vgg_panel_panel.append(row['image'])
        if ((row['prediction_class'] == '0') and (row['image'] not in vgg_panel_nopanel) and (row['label'] == '1')):
            vgg_panel_nopanel.append(row['image'])
        if ((row['prediction_class'] == '1') and (row['image'] not in vgg_nopanel_panel) and (row['label'] == '0')):
            vgg_nopanel_panel.append(row['image'])
        if ((row['prediction_class'] == '0') and (row['image'] not in vgg_nopanel_nopanel) and (row['label'] == '0')):
            vgg_nopanel_nopanel.append(row['image'])
            
csvfile.close()
print(len(vgg_panel_panel),len( vgg_panel_nopanel),len(vgg_nopanel_panel),len(vgg_nopanel_nopanel))


with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if ((row['lrpredict'] == '1') and (row['image'] not in lr_panel_panel) and (row['label'] == '1')):
            lr_panel_panel.append(row['image'])
        if ((row['lrpredict'] == '0') and (row['image'] not in lr_panel_nopanel) and (row['label'] == '1')):
            lr_panel_nopanel.append(row['image'])
        if ((row['lrpredict'] == '1') and (row['image'] not in lr_nopanel_panel) and (row['label'] == '0')):
            lr_nopanel_panel.append(row['image'])
        if ((row['lrpredict'] == '0') and (row['image'] not in lr_nopanel_nopanel) and (row['label'] == '0')):
            lr_nopanel_nopanel.append(row['image'])
            
csvfile.close()
print(len(lr_panel_panel),len( lr_panel_nopanel),len(lr_nopanel_panel),len(lr_nopanel_nopanel))