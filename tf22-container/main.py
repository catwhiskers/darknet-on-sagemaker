
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

from util import  get_dataset, AUTOTUNE, process_path, configure_for_performance
import util

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
    train_dir, validation_dir = get_train_val_dirs()
    train_dataset = get_dataset(train_dir)
    val_dataset = get_dataset(validation_dir)
    class_names = np.array(sorted([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"]))
    util.class_names = class_names 
    train_dataset = train_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    train_dataset = configure_for_performance(train_dataset)
    val_dataset = configure_for_performance(val_dataset)
    print(train_dataset)
    model = util.get_model()
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
    

if __name__ == "__main__":
    train()
