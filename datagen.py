from keras.preprocessing.image import ImageDataGenerator
from config import *

'''
amusement: 0; anger: 1; excitement: 2; sadness: 3
'''

pic_gen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data_generator = pic_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_cols, img_rows),
    batch_size=batch_size,
    class_mode='categorical'
)

test_data_generator = pic_gen.flow_from_directory(
    test_data_dir,
    target_size=(img_cols, img_rows),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_data_generator = pic_gen.flow_from_directory(
    validation_data_dir,
    target_size=(img_cols, img_rows),
    batch_size=batch_size,
    class_mode='categorical'
)
