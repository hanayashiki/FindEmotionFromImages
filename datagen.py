from keras.preprocessing.image import ImageDataGenerator
from config import *
from keras.applications.vgg16 import preprocess_input

'''
amusement: 0; anger: 1; excitement: 2; sadness: 3
'''

pic_gen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

train_data_generator = pic_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_cols, img_rows),
    batch_size=batch_size,
    class_mode='categorical',
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
