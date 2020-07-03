import numpy as np
import keras
from keras.models import Model, load_model

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16


IMAGE_SIZE = [224, 224]
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

prediction = Dense(2, activation="softmax")(x)

model2 = Model(inputs=vgg.input, outputs=prediction)

model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

def get_liveness_model():

    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (32, 32, 3)
    chanDim = -1

    # if we are using "channels first", update the input shape
    # and channels dimension
    if K.image_data_format() == "channels_first":
        inputShape = (3, 32, 32)
        chanDim = 1

    # first CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(16, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(2))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model


model = get_liveness_model()

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

model.summary()


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

shape = (224,224)

training_set = train_datagen.flow_from_directory(
    "C:/Summer Training/Face Recognition/Images/Train/",
    target_size=shape,
    batch_size=32,
    class_mode="categorical",
)

test_set = train_datagen.flow_from_directory(
    "C:/Summer Training/Face Recognition/Images/Test",
    target_size=shape,
    batch_size=32,
    class_mode="categorical",
)


r = model2.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
)

import matplotlib.pyplot as plt

plt.plot(r.history["loss"], label="train loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend()
plt.show()
plt.savefig("LossVal_loss")

# accuracies
plt.plot(r.history["acc"], label="train acc")
plt.plot(r.history["val_acc"], label="val acc")
plt.legend()
plt.show()
plt.savefig("AccVal_acc")

import tensorflow as tf

from keras.models import load_model

model.save("./Face Recognition/Liveness_model.h5")

import cv2

face_cascade = cv2.CascadeClassifier(
    "C:/Summer Training/Face Recognition/haarcascade_frontalface_default.xml"
)


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y : y + h, x : x + w]

    return cropped_face


from keras.preprocessing import image
from PIL import Image

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    # canvas = detect(gray, frame)
    # image, face =face_detector(frame)

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (32, 32))
        im = Image.fromarray(face, "RGB")

        img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)

        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)
        if pred[0][0] > 0.5:
            image = "Fake"
            cv2.putText(
                frame, image, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
            )
        if pred[0][1] > 0.5:
            image = "Real"
            cv2.putText(
                frame, image, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
            )
    else:
        cv2.putText(
            frame, "No Detection", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
        )

    cv2.imshow("Vide", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()
