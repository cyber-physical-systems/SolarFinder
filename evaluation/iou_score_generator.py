import numpy as np 
import cv2
import sys
import csv

import os.path as ospath

def hotkey():
    global outline_list
    global current_outline

    KEY_UNDO = ord('u')
    KEY_CLEAN = ord('c')
    KEY_NEXT = ord('n')
    KEY_SAVE = ord('s')
    KEY_QUIT = ord('q')

    key = cv2.waitKey(0)
    if key == KEY_QUIT:
        print('*** Quit')
        exit()
    else:
        print('*** Next Image')
        cv2.destroyAllWindows()

def main(argv):
    # print ('Number of arguments:', len(argv), 'arguments.')
    # print ('Argument List:', str(argv))
    contours_dir = "./data/panel/"
    rooftop_img_dir = "./panel/"
    rooftop_csv_path = './data/rooftop_solar_array_outlines_new.csv'
    rooftop_iou_csv_path = './rooftop_iou.csv'
    with open(rooftop_iou_csv_path, 'a') as csvfile:
        myFields = ['id', 'location_id', 'label', 'solar_list', 'contour_num','iou']
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        writer.writeheader()
    with open(rooftop_csv_path, newline='') as rooftop_csv_file:
        reader = csv.DictReader(rooftop_csv_file)
        for row in reader:
            roof = {}
            roof = row
            contour_mask = eval(row['contour_num'])
            # print(contour_mask)
            contour_img = np.zeros((800,800,3), np.uint8)
            for contour in contour_mask:
                contour_path =  contours_dir + contour + '.png'
                # print(contour_path )
                img = cv2.imread(contour_path)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                excluded_color = [0, 0, 0]
                indices_list = np.where(np.all(img != excluded_color, axis=-1))
                contour_img[indices_list] = [255, 255, 255]
            # cv2.imshow('img',contour_img)
            # cv2.waitKey(0)

            solar_mask = np.zeros((800,800,3), np.uint8)
            outline_list = eval(row['solar_list'])
            for outline in outline_list:
                # print(outline)
                pts = np.asarray(outline)
                cv2.fillPoly(solar_mask, np.int_([pts]), (255, 255, 255))
                # cv2.polylines(solar_mask, [pts], True, (0, 0, 255),  2)
            # cv2.imshow('img', solar_mask)
            # cv2.waitKey(0)
                # cv2.fillPoly(img_to_show, np.int_([pts]), (198, 133, 61))
                # cv2.fillPoly(img_to_show, np.int_([pts]), (255, 255, 255))
            #
            predict_gray_mask = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
            label_gray_mask = cv2.cvtColor(solar_mask, cv2.COLOR_BGR2GRAY)
            #
            # # rooftop_mask_size = cv2.countNonZero(rooftop_gray_mask)
            # # solar_mask_size = cv2.countNonZero(solar_gray_mask)
            # # size_ration = solar_mask_size / rooftop_mask_size
            # # print(rooftop_mask_size)
            # # print(solar_mask_size)
            # # print(size_ration)
            #
            # # IOU Score
            intersection = np.logical_and(predict_gray_mask, label_gray_mask)
            union = np.logical_or(predict_gray_mask, label_gray_mask)
            iou_score = np.sum(intersection) / np.sum(union)
            # print(iou_score)
            #
            # print(iou_score)
            #
            # # print(size_ration/iou_score)

            # cv2.imshow(row['id'], img_to_show)

            # hotkey()
            roof['iou'] = iou_score
            with open(rooftop_iou_csv_path, 'a') as csvfile_new:
                writer = csv.writer(csvfile_new)
                writer.writerow([roof['id'], roof['location_id'], roof['label'],
                                 roof['solar_list'], roof['contour_num'],roof['iou']])
            csvfile_new.close()
    
    rooftop_csv_file.close()   

if __name__ == "__main__":
    main(sys.argv[1:])