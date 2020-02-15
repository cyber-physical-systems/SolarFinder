import csv
csv_path = './rooftop_solar_array_outlines.csv'
csv_path_new = './rooftop_solar_array_outlines_new.csv'
csv_path_contour = './contour_all_positive.csv'
with open(csv_path_new, 'a') as csvfile:
    myFields = ['id', 'location','location_id','label','solar_list','contour_num']
    writer = csv.DictWriter(csvfile, fieldnames=myFields)
    writer.writeheader()
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        contour = row
        img_name = contour['id']
        contour_num = []
        with open(csv_path_contour, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if(row['image'] == img_name):
                    if (row['id']  not in contour_num):
                        contour_num.append(row['id'])
                else:
                    pass
            print(contour_num)
        csv_file.close()
        contour['contour_num'] = contour_num
        with open(csv_path_new, 'a') as csvfile_new:
            writer = csv.writer(csvfile_new)
            writer.writerow([contour['id'], contour['location'], contour['location_id'], contour['label'],
                             contour['solar_list'],contour['contour_num']])
        csvfile_new.close()
csvfile.close()


