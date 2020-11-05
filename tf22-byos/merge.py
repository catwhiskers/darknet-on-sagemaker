
import os 
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np 

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = [] 
def get_model():
    #data_augmentation = tf.keras.Sequential([
    #    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    #    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    #])
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
    x = preprocess_input(inputs)
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

import sys
import os  
import shutil
import subprocess
import json 
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import os
import tensorflow as tf
import pathlib 

#sagemaker data 
hyperparameters_file_path = "/opt/ml/input/config/hyperparameters.json"
inputdataconfig_file_path = "/opt/ml/input/config/inputdataconfig.json"
resource_file_path = "/opt/ml/input/config/resourceconfig.json"
data_files_path = "/opt/ml/input/data/"
failure_file_path = "/opt/ml/output/failure"
model_artifacts_path = "/opt/ml/model/"


#path configuration 
def get_train_val_dirs(): 
    train_dir = os.path.join(data_files_path, 'train')
    validation_dir = os.path.join(data_files_path, 'validation')

    train_dir = pathlib.Path(train_dir)
    validation_dir = pathlib.Path(validation_dir)
    return train_dir, validation_dir


import tarfile 
import shutil 
def train():
    global class_names
    train_dir, validation_dir = get_train_val_dirs()
    train_dataset = get_dataset(train_dir)
    val_dataset = get_dataset(validation_dir)
    class_names = np.array(sorted([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"]))
    train_dataset = train_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    train_dataset = configure_for_performance(train_dataset)
    val_dataset = configure_for_performance(val_dataset)
    print(train_dataset)
    model = get_model()
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    initial_epochs = 10
    history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=val_dataset)
    model.save('./tlmodel') 
    tar = tarfile.open("tlmodel.tar.gz", "w:gz")
    tar.add("./tlmodel", arcname="tlmodel")
    tar.close() 
    shutil.move("tlmodel.tar.gz", model_artifacts_path)
    
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    print(args.train)
    #train(args)
    train()
