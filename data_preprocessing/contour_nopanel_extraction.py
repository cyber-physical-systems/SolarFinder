import csv
import cv2
csvpath = '/aul/homes/final_contour/house3' + '/nopanelcontour_features.csv'
with open(csvpath, 'a') as updatecsv:
    myFields = ['id', 'image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
    writer = csv.DictWriter(updatecsv, fieldnames=myFields)
    writer.writeheader()
updatecsv.close()
csv_path = '/aul/homes/data/house3/contour_features.csv'
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (row['label']==str(0)):
            contour = row
            img_path =  '/aul/homes/data/house3/' + 'contour_all/' + row['id'] + '.png'
            img = cv2.imread(img_path)
            img_newpath = '/aul/homes/final_contour/house3/nopanel/' + row['id'] + '.png'
            cv2.imwrite(img_newpath ,img)
            print(contour['id'])
            with open(csvpath, 'a') as updatecsv:
                writer = csv.writer(updatecsv)
                writer.writerow([contour['id'], contour['image'],contour['size'],contour['pole'],contour['mean'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
            updatecsv.close()                           
csvfile.close()
print('finish')