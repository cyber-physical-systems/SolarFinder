import numpy as np 
import cv2
import sys
import csv

import os.path as path

outline_list = []
current_outline = []
current_row = {}

def save_solar_polygon(solar_polygon):
    global output_csv_path
    global current_row
    print('polygon to save: ' + str(solar_polygon))
    with open(output_csv_path, 'a') as output_csv_file:
        output_row = {}
        # roof_id, outline_nodes, parameters, groundtruth, solar_outline
        output_row['roof_id'] = current_row['roof_id']
        output_row['outline_nodes'] = current_row['outline_nodes']
        output_row['parameters'] = current_row['parameters']
        output_row['groundtruth'] = current_row['groundtruth']
        output_row['solar_outline'] = solar_polygon

        writer = csv.writer(output_csv_file)
        writer.writerow([output_row['roof_id'], output_row['outline_nodes'], output_row['parameters'], output_row['groundtruth'], output_row['solar_outline']])
    output_csv_file.close()
    print('saved')

def hotkey():
    global tile_id
    global outline_list
    global current_outline
    global current_row
    global total_labeled_counter

    KEY_TRUE = ord('t')
    KEY_FALSE = ord('f')
    KEY_UNDO = ord('u')
    KEY_CLEAN = ord('c')
    KEY_NEXT = ord('n')
    KEY_SAVE = ord('s')
    KEY_QUIT = ord('q')
    KEY_INFO = ord('i')

    key = cv2.waitKey(0)
    if key == KEY_TRUE:
        current_row['groundtruth'] = True
        print('*** TRUE: has solar array')
        hotkey()
    elif key == KEY_FALSE:
        current_row['groundtruth'] = False
        print('*** FALSE: has NO solar array')
        hotkey()
    elif key == KEY_UNDO:
        print('*** Undo')
        if len(current_outline) >= 1:
            del current_outline[-1]
        print(current_outline)
        hotkey()
    elif key == KEY_CLEAN:
        print('*** Clean')
        current_outline = []
        print(current_outline)
        hotkey()
    elif key == KEY_NEXT:
        if len(current_outline) > 0:
            outline_list.append(current_outline)
        print(outline_list)
        print('*** Mark next outline')
        current_outline = []
        print(current_outline)
        hotkey()
    elif key == KEY_SAVE:
        print('*** Save')
        total_labeled_counter += 1
        if len(current_outline) > 0:
            outline_list.append(current_outline)
            current_outline = []
        # if len(outline_list) <= 0:
        #     outline_list
        if len(outline_list) > 0:
            current_row['groundtruth'] = True
        save_solar_polygon(outline_list)
        cv2.destroyAllWindows()
    elif key == KEY_QUIT:
        print(f'Last unfinished tile: {tile_id}')
        print(f'You have totally labeled {total_labeled_counter} roofs this time.')
        print('*** Quit solar marker')
        exit()
    elif key == KEY_INFO:
        print('*** Current Rooftop Info:')
        print(current_row)
        hotkey()
    else:
        print('*** Undefined key')
        hotkey()

def onMouse(event, x, y, flags, param):
    global current_outline
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        current_outline.append(click_point)
        print('*** Add point: ' + str(click_point))
        print(current_outline)

def main(argv):
    # print ('Number of arguments:', len(argv), 'arguments.')
    
    global outline_list
    global current_outline
    global output_csv_path
    global current_row
    global tile_id
    global total_labeled_counter

    tile_id = str(argv[0])

    if not(int(tile_id) >= 0 and int(tile_id) < 6400):
        print('Error: Tile id is out of range.')
        exit()

    workspace_path = ''

    mass_osm_path = f'{workspace_path}data/Mass_osm_information.csv'

    mass_osm_reader = None

    total_labeled_counter = 0

    with open(mass_osm_path, newline='') as mass_osm_file:
        mass_osm_reader = csv.DictReader(mass_osm_file)

        for row in mass_osm_reader:
            number_of_roof = int(row['number_roof'])

            current_mass_osm_tile_id = row['id']

            if int(current_mass_osm_tile_id) < int(tile_id) or number_of_roof <= 0:
                continue
            else:
                tile_id = current_mass_osm_tile_id
            

            print(f'*** Start Labelling tile: {str(tile_id)} ***')


            output_csv_path = f'{workspace_path}output/{tile_id}_groundtruth.csv'

            rooftop_img_dir = f'{workspace_path}data/rooftop/{tile_id}/'

            rooftop_csv_dir = f'{workspace_path}data/osm_parsed/'
            rooftop_csv_path = f'{rooftop_csv_dir}{tile_id}.csv'

            if not path.exists(rooftop_csv_path):
                # print(rooftop_csv_path)
                print(f'Tile {tile_id} csv file does not exist.')
                exit()

            last_row_of_output = None

            continued_work = False
            skipped = False
            
            if path.exists(output_csv_path):
                continued_work = True
            
            # Get las row of output csv file
            if continued_work:
                with open(output_csv_path, newline='') as output_csv_file:
                    reader = csv.DictReader(output_csv_file)
                    for row in reader:
                        last_row_of_output = row['roof_id']
                output_csv_file.close()
                print('This is a continued work.')

            if not continued_work:
                output_csv_header = ['roof_id', 'outline_nodes', 'parameters', 'groundtruth', 'solar_outline']
                with open(output_csv_path, 'a') as output_csv_file:
                    writer = csv.DictWriter(output_csv_file, fieldnames=output_csv_header)
                    writer.writeheader()
                output_csv_file.close()

            print(f'Start labelling tile: {rooftop_csv_path}.')
            roof_counter_of_current_tile = 0
            with open(rooftop_csv_path, newline='') as rooftop_csv_file:
                reader = csv.DictReader(rooftop_csv_file)

                for row in reader:
                    roof_counter_of_current_tile += 1

                    if continued_work and last_row_of_output is not None:
                        if not skipped:
                            if last_row_of_output != row['roof_id']:
                                continue
                            elif last_row_of_output == row['roof_id']:
                                skipped = True
                                continue

                    outline_list = []
                    current_outline = []

                    current_row = row
                    current_row['groundtruth'] = False

                    rooftop_img_file_name = row['roof_id'] + '.png'
                    print(f'Labelling {rooftop_img_file_name}... ({roof_counter_of_current_tile}/{number_of_roof})')
                    img_path = rooftop_img_dir + rooftop_img_file_name
                    
                    if not path.exists(img_path):
                        print(f'Error: Rooftop image {img_path} is not exist!')
                        exit()
                    
                    img = cv2.imread(img_path)
                    window_name = row['roof_id']
                    cv2.namedWindow(window_name)
                    cv2.moveWindow(window_name, 20, 20)
                    cv2.imshow(window_name, img)

                    cv2.setMouseCallback(row['roof_id'], onMouse)

                    hotkey()
            
            rooftop_csv_file.close()
        mass_osm_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])