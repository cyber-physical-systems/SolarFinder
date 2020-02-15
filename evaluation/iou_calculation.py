import csv
import math
num = {}
for i in range(0,11):
    num[i] = 0
number = 0
csv_path = './rooftop_iou.csv'
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        iou = float(row['iou'])
        for i in range(0,11):
            if (iou > i*0.1):
                num[i] = num[i] +1
        number = number + 1
csvfile.close()
print(num)
print(number)
