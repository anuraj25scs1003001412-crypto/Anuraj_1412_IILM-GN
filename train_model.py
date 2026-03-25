import cv2
import os
import numpy as np

faces = []
labels = []
label_map = {}
current_id = 0

data_dir = "data"
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for name in os.listdir(data_dir):
    folder = os.path.join(data_dir, name)
    if not os.path.isdir(folder):
        continue
    label_map[current_id] = name

    for img_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
        face = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in face:
            faces.append(img[y:y+h, x:x+w])
            labels.append(current_id)
    current_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.write("trainer.yml")

with open("labels.txt", "w") as f:
    for id, name in label_map.items():
        f.write(f"{id},{name}\n")

print("Training completed successfully!")