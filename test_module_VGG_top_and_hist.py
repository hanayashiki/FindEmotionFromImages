import keras
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import traceback
import os
from mAP import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import *
from datagen import *
from single_predict import *
from rgb2hsv import *

base_model = VGG16(weights='imagenet', include_top=True)
hist = Histogram(base_model.input)
dense = Dense(30, activation='relu', name='connect_basemodel')(hist)
concat = Concatenate()([dense, base_model.output])
prediction = Dense(img_category_count, activation='softmax', name='prediction')(concat)

model_weight = 'vgg_top_4cat_hist_preprocessor.h5'

model = Model(inputs=base_model.input, outputs=prediction)
try:
    if os.path.exists(checkpoints_dir + model_weight):
        model.load_weights(checkpoints_dir + model_weight)
        print("Load old weights. ")
    else:
        print("Use new weights.")
except:
    print("File error. Use new weights.")

# freeze vgg16 weights
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', precision_0, precision_1, precision_2, precision_3, precision])

print(img_predict(model, 'test_single_pic\\amusement746.jpg'))

# model.fit_generator(
#     train_data_generator,
#     steps_per_epoch=steps_per_epoch_train,
#     epochs=epoch,
#     validation_data=test_data_generator,
#     validation_steps=steps_per_epoch_test,
#     callbacks=[ModelCheckpoint(
#         filepath=checkpoints_dir + model_weight,
#         save_best_only=True,
#         save_weights_only=True
#     )]
# )