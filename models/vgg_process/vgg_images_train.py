# this is used to trian the vgg model to classify panel and nopanel
import keras
import numpy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import keras
from keras import models
from keras import layers
from keras.callbacks import TensorBoard
from keras.applications import VGG16
import datetime
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# data to train vgg model

# Input dirs

workspace_dir = './dataset'

original_dataset_dir = os.path.join(workspace_dir, 'contours')

train_dir = os.path.join(original_dataset_dir, 'train')

validation_dir = os.path.join(original_dataset_dir, 'validation')

train_panel_dir = os.path.join(train_dir, 'panel')

train_nopanel_dir = os.path.join(train_dir, 'nopanel')

validation_panel_dir = os.path.join(validation_dir, 'panel')

validation_nopanel_dir = os.path.join(validation_dir, 'nopanel')

# Output dirs

training_model_output_dir = './solar_panel/smalldata/'

training_log_dir = './solar_panel/smalldata/'

model_output_dir = './solar_panel/smalldata/'

# pretrained model imagenet
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

NAME = "VGG-16_pretrain_1"
print(NAME)

# add the last sequential 
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = True

set_trainable = False

print('trainable weights is :', len(model.trainable_weights))

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')


# model.compile(loss='binary_crossentropy',  optimizer=optimizers.RMSprop(lr=2e-5), )
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
# use checkpointer to stop trainnig early 
checkpointer = ModelCheckpoint(filepath = training_model_output_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
log_dir = training_log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print (log_dir)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
callbacks = [ checkpointer,earlystopper,tensorboard_callback]

history = model.fit_generator(
    train_generator,
    samples_per_epoch=1000,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2,
    callbacks=callbacks)
path = model_output_dir
model.save(os.path.join(path ,'VGG16_pretrain_all.model'))

print('finish')

sys.stdout.flush()





