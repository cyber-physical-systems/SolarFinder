import csv
import math
csv_path = './orientation_positive.csv'
num_all = 0
num_5 = 0
num_10 = 0
num_15 = 0
num_20 = 0
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        contour_orientation =float(row['contour'])
        roof_orientation = float(row['roof'])
        contour_orientation_45differ  =  math.fabs(math.fabs(contour_orientation)- 45)
        roof_orientation_45differ = math.fabs(math.fabs(roof_orientation)- 45)
        differ =  math.fabs(contour_orientation_45differ - roof_orientation_45differ)
        if(differ < 5):
            num_5 = num_5 + 1
        if (differ < 10):
            num_10 = num_10 + 1
        if (differ < 15):
            num_15 = num_15 + 1
        if (differ < 20):
            num_20 = num_20 + 1
        num_all = num_all + 1
csvfile.close()
percent_5 = num_5 /num_all
percent_10 = num_10 /num_all
percent_15 = num_15 /num_all
percent_20 = num_20 /num_all

print(num_all ,num_5,num_10,num_15,num_20)
print(percent_5,percent_10,percent_15,percent_20)