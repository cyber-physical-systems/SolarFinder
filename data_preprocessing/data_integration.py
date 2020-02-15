import csv
import cv2


csvpath_all = '/aul/homes/final/split/location810/contour_all.csv'
with open(csvpath_all, 'a') as csvfile:
    myFields = ['id', 'location','image', 'size','pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()
    
csvpath_yes = '/aul/homes/final/split/location810/contour_features.csv'
with open(csvpath_yes, 'a') as csvfile:
    myFields = ['id', 'location', 'image', 'size','pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()
    
csvpath_no = '/aul/homes/final/split/location810/no_contour_features.csv'
with open(csvpath_no, 'a') as csvfile:
    myFields = ['id',  'location','image', 'size','pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()
    
for i in range(8,11):
    csv_path = '/aul/homes/final_contour/house' + str(i) + '/contour_all.csv'
    with open(csv_path, newline='') as csv_file:  
        reader = csv.DictReader(csv_file)
        for row in reader:
            contour = {}
            contour = row
            contour['location'] = 'location' + str(i)
            with open(csvpath_all, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([contour['id'], contour['location'],contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
            csvfile.close()
#             print(contour)
            if(contour['label'] == str(1)):
                with open(csvpath_yes, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([contour['id'], contour['location'],contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
                csvfile.close()
                
            if(contour['label'] == str(0)):
                with open(csvpath_no, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([contour['id'], contour['location'],contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
                csvfile.close()
                
           
    csv_file.close()
    print(csv_path)

    
    
    
    
import csv
import cv2


csvpath_train_nopanel = '/aul/homes/final/nosplit/train/train_nopanel.csv'
with open(csvpath_train_nopanel, 'a') as csvfile:
    myFields = ['id', 'location','image', 'size','pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()
    
csvpath_test_nopanel = '/aul/homes/final/nosplit/test/test_nopanel.csv'
with open(csvpath_test_nopanel , 'a') as csvfile:
    myFields = ['id', 'location', 'image', 'size','pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()
    
csvpath_validation_nopanel = '/aul/homes/final/nosplit/validation/validation_nopanel.csv'
with open(csvpath_validation_nopanel, 'a') as csvfile:
    myFields = ['id',  'location','image', 'size','pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()
    

csv_path = '/aul/homes/final/nosplit/no_contour_features.csv'
i = 0 
with open(csv_path, newline='') as csv_file:  
    reader = csv.DictReader(csv_file)
    for row in reader:
        contour = {}
        contour = row
        if ((i %10) <3):
            with open(csvpath_test_nopanel, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([contour['id'], contour['location'],contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
            csvfile.close()
        
#             print(contour)
        elif ((i %10) >7):
            with open(csvpath_validation_nopanel, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([contour['id'], contour['location'],contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
            csvfile.close()

        else:
            with open(csvpath_train_nopanel, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([contour['id'], contour['location'],contour['image'],contour['size'],contour['pole'],contour['mean'],contour['stddev'],contour['square'],contour['ratiowh'],contour['ratioarea'],contour['approxlen'],contour['numangle'],contour['numangle90'], contour['numangle70'],contour['label']])
            csvfile.close()
        i = i + 1        
           
csv_file.close()
            
            
            