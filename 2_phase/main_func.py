import cv2
import numpy as np
import keras
from PIL import Image
from utils import *
import os


model_unet = load_model_weight('files/model_final.hdf5')
model_classification = keras.models.load_model("files/model_cnn_classify.h5")


def unet(img):
    x = cv2.imread(img)
    x = cv2.resize(x, (256, 256))
    x = np.clip(x - np.median(x)+127, 0, 255)
    x = x/255.0
    x = x.astype(np.float32)
    x = x.reshape(1, 256, 256, 3)
    
    y_pred = parse(model_unet.predict(x)[0][..., -1])

    # Convert to 3d black and white image
    mask = mask_to_3d(y_pred) * 255.0

    return mask

def classification(img_path, mask):
    infected = False
    pred_list = []
    cell_list = []

    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(mask, kernel, iterations=1)

    result = cv2.bitwise_and(img, img, mask=mask)

    contours, _ = cv2.findContours(
        img_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for (i, c) in enumerate(contours):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)

        # Put into list
        crop = result[y:y+h+10, x:x+w+10]
        cell_list.append(crop)

    for cell in cell_list:
        image = cv2.resize(cell, (64, 64))
        image = image.reshape(1, 64, 64, 3)

        prediction = model_classification.predict(np.array(image))
        pred_list.append(np.argmax(prediction))

    pred_list.sort(reverse=True)
    infected = True if pred_list[0] > 0 else infected
    return infected

def highlight_cell(img_path,mask):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(256,256))
    img_org = cv2.resize(img,(256,256))
    mask = mask.astype('uint8')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    res = cv2.drawContours(img, contours, -1, (0,255,0), 1)
    
    return [Image.fromarray(img_org), Image.fromarray(res)]

def predict(image_path):
    mask = unet(image_path)
    pred = classification(image_path, mask)
    
    img_list = highlight_cell(image_path,mask)
    
    if pred:
        return pred, img_list[1]
    else:
        return pred, img_list[0]