
import os 
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np 

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = [] 
def get_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    IMG_SHAPE = IMG_SIZE + (3,)

    base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    base_model.build((None,)+IMG_SHAPE)
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model 


def get_label(file_path):
  # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = class_names == parts[-2]
    one_hot = tf.dtypes.cast(one_hot, tf.int32)  
  # Integer encode the label
    return tf.cast(tf.argmax(one_hot), tf.int64) 


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE[1], IMG_SIZE[0]])
    return img

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def get_dataset(data_dir): 
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    return list_ds

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
