import csv
import os.path
from os import path

csv_path = '/aul/homes/dataset/dataset930/house' + str(1) + '/house' + str(1) + '.csv'
csvpath = '/aul/homes//dataset/dataset930/house' + str(1) + '/location' + str(1) + '.csv'

with open(csvpath, 'a') as csvupdate:
    myFields = ['id', 'location','label']
    writer = csv.DictWriter(csvupdate, fieldnames=myFields)
    writer.writeheader()
csvupdate.close()
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        img = {}
        img['id'] = row['id']
        img['location'] = row['location']
        img['lable'] = row['label']
        if (path.exists('/aul/homes/dataset/dataset930/house' + str(1) + '/roof/'  + img['id'] + '.png') == True):
            with open(csvpath, 'a') as csvupdate:
                writer = csv.writer(csvupdate)
                writer.writerow([img['id'],img['location'], img['lable']])
                csvupdate.close()
csvfile.close()
print('finish')
           
            
        
