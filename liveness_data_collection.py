import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(
    "C:/Summer Training/Face Recognition/haarcascade_frontalface_default.xml"
)


def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.1, 5)

    if faces is ():
        return None
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y : y + h + 50, x : x + w + 50]

    return cropped_face


print("Enter Real or Fake")
x = input()
print("Enter a Name")
y = input()

if x == "r":
    path_train = os.path.join(
        "C:/Summer Training/Face Recognition/Images/Train/Real/", y
    )
    path_test = os.path.join("C:/Summer Training/Face Recognition/Images/Test/Real/", y)
elif x == "f":
    path_train = os.path.join(
        "C:/Summer Training/Face Recognition/Images/Train/Fake/", y
    )
    path_test = os.path.join("C:/Summer Training/Face Recognition/Images/Test/Fake/", y)

if not os.path.exists(path_train):
    os.mkdir(path_train)
if not os.path.exists(path_test):
    os.mkdir(path_test)
else:
    print("Folder already exists")

cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        file_name_path = path_train + "/" + y + str(count) + ".jpg"
        if count >= 70:
            file_name_path = path_test + "/" + y + str(count) + ".jpg"
        cv2.imwrite(file_name_path, face)
        cv2.putText(
            face, str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        cv2.imshow("Face Cropper", face)

    else:
        print("Face not found")
        pass
    if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
print("DONE")

