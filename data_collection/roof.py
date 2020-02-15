# this is used to download  the roof images from google static map ,
# we download the original images and mask from google static map (free) then use and operation to get the roof ROI,
# so we can process and label the roof images
import glob as gb
import json
import os

import cv2
import numpy as np
import requests

i = 0
json_path = gb.glob("./10house/house1/map.json")
for file in json_path:
    with open(file, 'r') as file:
        urls = json.load(file)
        for url in urls:
            i = i + 1
            id = url['id']
            mask = url['mask']
            image = url['image']
            mask = requests.get(mask)
            image = requests.get(image)
            fmask = open(os.path.join('./10house/house1/image/', format(str('1')) + '.png'), 'ab')
            fimg = open(os.path.join('./10house/house1/mask/', format(str('1')) + '.png'), 'ab')
            fmask.write(mask.content)
            fimg.write(image.content)
            fmask.close()
            fimg.close()
            tag = cv2.imread(os.path.join('./10house/house1/image/', format('1') + '.png'))
            real = cv2.imread(os.path.join('./10house/house1/mask/', format('1') + '.png'))
            lower = np.array([0, 0, 100])
            upper = np.array([40, 40, 255])
            img = cv2.inRange(tag, lower, upper)

            # and operations with images
            img = np.expand_dims(img, axis=2)
            img = np.concatenate((img, img, img), axis=-1)
            result = cv2.bitwise_and(real, img)
            cv2.imwrite(os.path.join('./10house/house1/roof/' + format(str(id)) + '.png'), result)
            os.remove("./10house/house1/image/1.png")
            os.remove("./10house/house1/mask/1.png")
