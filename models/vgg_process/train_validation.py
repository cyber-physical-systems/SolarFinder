import csv
import cv2

csvpath_all = './feature_test.csv'
with open(csvpath_all, 'a') as csvfile:
    myFields = ['id','image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction','prediction_class','label']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()

csv_path = './vgg_predict-Copy2.csv'
with open(csv_path, newline='') as csv_file:  
    reader = csv.DictReader(csv_file)
    for row in reader:
        contour = row
        with open(csvpath_all, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([contour['id'],contour['image'],contour['size'],contour['pole'],contour['mean'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['prediction'],contour['prediction_class'],contour['label']])
        csvfile.close()


            
            
            
            