import sys
import csv

def main(argv):
    # print ('Number of arguments:', len(argv), 'arguments.')
    # print ('Argument List:', str(argv))

    output_dir = '/Users/Aaron/projects/solarpanel/data/rooftop_with_solar_array/'
    output_csv_path = output_dir + 'rooftop_positive_dataset.csv'
    output_csv_header = ['id', 'location', 'location_id', 'label']

    with open(output_csv_path, 'a') as output_csv_file:
        writer = csv.DictWriter(output_csv_file, fieldnames=output_csv_header)
        writer.writeheader()
    output_csv_file.close()

    origin_rooftop_csv_dir = '/Users/Aaron/projects/solarpanel/data/origin_rooftop_csv/'
    
    for location_id in range(1,11):
        current_origin_rooftop_csv_path = origin_rooftop_csv_dir + 'house' + str(location_id) + '.csv'
        print(current_origin_rooftop_csv_path)
        
        with open(current_origin_rooftop_csv_path, newline='') as current_origin_rooftop_csv_file:
            reader = csv.DictReader(current_origin_rooftop_csv_file)
            for row in reader:
                if row['label'] == '0':
                    continue
                output_row = {}
                output_row['id'] = row['id']
                output_row['location'] = row['location']
                output_row['location_id'] = str(location_id)
                output_row['label'] = row['label']

                with open(output_csv_path, 'a') as output_csv_file:
                    writer = csv.writer(output_csv_file)
                    writer.writerow([output_row['id'], output_row['location'], output_row['location_id'], output_row['label']])
                output_csv_file.close()
        
        current_origin_rooftop_csv_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
