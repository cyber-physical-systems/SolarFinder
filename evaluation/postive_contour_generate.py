
import csv
csv_path ='./feature_test_all_vgg_svm_linear.csv'
csv_path_new = './contour_all_positive.csv'
with open(csv_path_new, 'a') as csvfile:
    myFields = ['id', 'location','image', 'label','predict']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
csvfile.close()
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        contour = row
        if(contour['linear_nosplit_class']== '1'):
            with open(csv_path_new , 'a') as csvfile_new:
                writer = csv.writer(csvfile_new)
                writer.writerow([contour['id'], contour['location'], contour['image'],contour['label'], contour['linear_nosplit_class']])
            csvfile_new.close()
csvfile.close()
