from PIL import Image
import os
import numpy as np
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
import keras.utils as image
from tqdm import tqdm


ResNet_model = Sequential()
ResNet_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
ResNet_model.add(Dense(133, activation='softmax'))
ResNet_model.load_weights('../saved_models/weights.best.ResNet.hdf5')
face_cascade = cv2.CascadeClassifier(
    '../haarcascades/haarcascade_frontalface_alt.xml')
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
dog_names = [item[20:-1]
             for item in sorted(glob("../data/dog_images/train/*/"))]


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path)
                       for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def face_detector(img_path):
    """# returns "True" if face is detected in image stored at img_path"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    """
    ### returns "True" if a dog is detected in the image stored at img_path
    """
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def extract_Resnet50(tensor):
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def resnet_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = ResNet_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def dog_breed_detector(img_path):
    prediction = resnet_predict_breed(img_path)
    predict_name = os.path.basename(prediction).split(".")[1].replace("_", " ")
    predict_folder = np.array(
        glob("../data/dog_images/train/" + os.path.basename(prediction) + "/*"))[0]
    predict_message = ''
    # Original
    base_name = f"Original: {os.path.basename(img_path)}"
    img_base = Image.open(img_path)
    # Predict
    not_dog_n_human = False
    if dog_detector(img_path):
        predict_message = f"Predict: {predict_name}"
    elif face_detector(img_path) > 0:
        predict_message = f"Human: look like a {predict_name}"
    else:
        predict_message = "Predict: Neither dog or human!"
        not_dog_n_human = True
    try:
        predict_img = Image.open(predict_folder)
        if not_dog_n_human:
            predict_img = img_base
    except Exception as e:
        print(e)
    return img_base, base_name, predict_img, predict_message


def plot_img():
    img_files = np.array(glob("uploads/*"))
    fig = plt.figure(figsize=(9, len(img_files)*4))
    columns = 2
    rows = len(img_files)
    ax = []
    img_l, message_l = [], []
    for img in img_files:
        img_base, base_name, predict_img, predict_message = dog_breed_detector(
            img)
        img_l.append(img_base)
        img_l.append(predict_img)
        message_l.append(base_name)
        message_l.append(predict_message)

    for i in range(columns*rows):
        ax.append(fig.add_subplot(rows, columns, i+1))
        ax[-1].set_title(message_l[i])
        plt.imshow(img_l[i],)
    return fig


if __name__ == '__main__':
    plot_img()
