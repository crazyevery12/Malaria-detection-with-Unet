
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from keras.models import load_model
from glob import glob
from utils import *
from metrics import *
from model_rec.model_larger_fillter_extracConvo import build_model


def read_image(x):
    x = x.decode()
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    return image


def read_mask(y):
    y = y.decode()
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))
    mask = mask/255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def parse_data(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        y = np.concatenate([y, y], axis=-1)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 2])
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.map(map_func=parse_data)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch)
    return dataset


if __name__ == "__main__":

    tf.config.run_functions_eagerly(True)

# rest of the code

    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("files")

    train_path = "data/train"
    valid_path = "data/valid"

    # Training

    train_x = sorted(glob(os.path.join(train_path, "img", "*")))
    train_y = sorted(glob(os.path.join(train_path, "mask", "*")))

    # Shuffling
    train_x, train_y = shuffling(train_x, train_y)

    # Validation
    valid_x = sorted(glob(os.path.join(valid_path, "img", "*")))
    valid_y = sorted(glob(os.path.join(valid_path, "mask", "*")))

    batch_size = 16
    epochs = 50

    shape = (256, 256, 3)

    model = build_model(shape)
    lr = 0.001
    csv_log = "files/data_final_largefilter_extraConvo.csv"
    checkpoint_name = "files/final_checkpoint_extraConvo.hdf5"

    metrics = [
        dice_coef,
        iou,
        Recall(),
        Precision()
    ]

    model.compile(loss=dice_loss, optimizer=Adam(lr),
                  metrics=metrics, run_eagerly=True)

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    callbacks = [
        ModelCheckpoint(checkpoint_name),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50),
        CSVLogger(csv_log),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50,
                      restore_best_weights=False)
    ]

    train_steps = (len(train_x)//batch_size)
    valid_steps = (len(valid_x)//batch_size)

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model.fit(train_dataset,
              epochs=50,
              validation_data=valid_dataset,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              callbacks=callbacks,
              shuffle=False)

    model.save("files/model_large_filter_extraConvo_2.hdf5")
