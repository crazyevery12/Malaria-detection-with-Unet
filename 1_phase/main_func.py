from tensorflow.keras.applications import VGG19
import cv2
import pickle



SIZE = 256
img_path = 'data/malaria/190813120342.png'
svm_model = 'files/finalized_svm_model.sav'

svm_model = pickle.load(open(svm_model, 'rb'))
VGG_model = VGG19(weights='imagenet', include_top=False,
                  input_shape=(SIZE, SIZE, 3))
for layer in VGG_model.layers:
    layer.trainable = False


def classify(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = img.reshape(1, 256, 256, 3)

    feature = VGG_model.predict(img)
    feature = feature.reshape(feature.shape[0], -1)

    prediction = svm_model.predict(feature)
    return prediction[0]

pred = classify(img_path)
print(pred)