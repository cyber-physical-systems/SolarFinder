import csv
import cv2

num = 0
csv_path = '/aul/homes/final/nosplit/train/train_nopanel.csv'
csvpath_train_nopanel = '/aul/homes/final/nosplit/train/train_nopanel.csv'
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
#         if (row['label']==str(1)):
#             contour = row
        img_name = row['id']
        location = row['location'][-1]
        if (location == '0'):
            print(row['locaiton']
        
#         print(location)
        img_path =  '/aul/homes/final_contour/house' + str('location') + '/panel/' + 'img_name' + '.png''
        img = cv2.imread(img_path)
        img_newpath = '/aul/homes/final/split/train/nopanel/' +  'img_name' + '.png'
        cv2.imwrite(img_newpath ,img)
csvfile.close()        
       