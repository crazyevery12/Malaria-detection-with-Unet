import numpy as np 
import cv2
from PIL import Image
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG19
from sklearn import svm
import pickle


SIZE = 256
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


# Data for training
image_directory = 'data/'
train_images = []
train_labels = [] 

parasitized_images = os.listdir(image_directory + 'malaria/')
# Go thru every image and resize + add label.
for i, image_name in enumerate(parasitized_images):

    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'malaria/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        train_images.append(np.array(image))
        train_labels.append(1)
        
uninfected_images = os.listdir(image_directory + 'no_malaria/')
for i, image_name in enumerate(uninfected_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'no_malaria/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        train_images.append(np.array(image))
        train_labels.append(0)
        
X_train, X_test, y_train, y_test = train_test_split(train_images, np.array(train_labels),
                                                    test_size=0.20, random_state=42)

# Define VGG19 and set not to update trainable param
VGG_model = VGG19(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
for layer in VGG_model.layers:
	layer.trainable = False

# Extract feature and reshape to mathch svm input
feature_extractor=VGG_model.predict(np.array(X_train))
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

# Define and train
svm_model = svm.SVC(C=0.1, gamma=1, kernel='linear', probability=True)
svm_model.fit(features,y_train)

# Save trained svm
file = 'files/finalized_svm_model.sav'
pickle.dump(svm_model, open(file, 'wb'))