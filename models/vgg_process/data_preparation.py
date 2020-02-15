# Create  dataset for panel and nopanel
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob as gb
original_dataset_dir = './location17/'

base_dir = './split/'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Create directories
train_dir = os.path.join(base_dir,'train/')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation/')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
# test_dir = os.path.join(base_dir,'test/')
# if not os.path.exists(test_dir):
#     os.mkdir(test_dir)

train_panel_dir = os.path.join(train_dir,'panel/')
if not os.path.exists(train_panel_dir):
    os.mkdir(train_panel_dir)

train_nopanel_dir = os.path.join(train_dir,'nopanel/')
if not os.path.exists(train_nopanel_dir):
    os.mkdir(train_nopanel_dir)

validation_panel_dir = os.path.join(validation_dir,'panel/')
if not os.path.exists(validation_panel_dir):
    os.mkdir(validation_panel_dir)

validation_nopanel_dir = os.path.join(validation_dir, 'nopanel/')
if not os.path.exists(validation_nopanel_dir):
    os.mkdir(validation_nopanel_dir)


num = 0
img_path = gb.glob("./panel_samesize/*.png")
for path in img_path:
    img_name = path.split("/")[-1]
    
    img = cv2.imread(path)
#     0,1,2,3,4,5,6,
    if ((num % 10) < 7):
        cv2.imwrite(os.path.join(train_panel_dir  + img_name),img)
#     elif ((num % 10) > 6):
#         pass
#         cv2.imwrite(os.path.join(test_panel_dir +str(1) + img_name),img)
    else:
        cv2.imwrite(os.path.join(validation_panel_dir + img_name),img)
    num = num + 1 
num = 0 
img_path = gb.glob("./nopanel_undersample/*.png")
for path in img_path:
    img_name = path.split("/")[-1]
   
    img = cv2.imread(path)
    if ((num % 10) < 7):
        cv2.imwrite(os.path.join(train_nopanel_dir +img_name),img)
#     elif ((num % 10) > 6):
#         cv2.imwrite(os.path.join(test_nopanel_dir +img_name),img)
    else:
        cv2.imwrite(os.path.join(validation_nopanel_dir +img_name),img)
    num = num + 1                                            
# Sanity checks
print('total training panel images:', len(os.listdir(train_panel_dir)))
print('total training nopanel images:', len(os.listdir(train_nopanel_dir)))
print('total validation panel images:', len(os.listdir(validation_panel_dir)))
print('total validation nopanel images:', len(os.listdir(validation_nopanel_dir)))
