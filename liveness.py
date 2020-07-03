import cv2
from keras.models import load_model
import numpy as np

face_cascade = cv2.CascadeClassifier(
    "C:/Summer Training/Face Recognition/haarcascade_frontalface_default.xml"
)
import h5py

f = h5py.File("C:/Summer Training/Face Recognition/livenessvgg.h5", "r+")
data_p = f.attrs["training_config"]
data_p = data_p.decode().replace("learning_rate", "lr").encode()
f.attrs["training_config"] = data_p
f.close()

model = load_model("C:/Summer Training/Face Recognition/livenessvgg.h5")


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
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, "RGB")

        img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)

        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)
        if pred[0][0] > 0.7:
            image = "Fake"
            cv2.putText(
                frame, image, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
            )
        if pred[0][1] > 0.7:
            image = "Real"
            cv2.putText(
                frame, image, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
            )
    else:
        cv2.putText(
            frame, "Wait", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
        )

    cv2.imshow("Vide", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()
