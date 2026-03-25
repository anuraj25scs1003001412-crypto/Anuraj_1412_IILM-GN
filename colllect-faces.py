import cv2
import os

data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

name = input("Enter person name: ")  
save_dir = os.path.join(data_dir, name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
target_count = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = gray[y:y+h, x:x+w]

        # save face
        img_path = os.path.join(save_dir, f"{name}_{count}.jpg")
        cv2.imwrite(img_path, face_roi)
        count += 1
        print(f"Saved image {count} for {name}")

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= target_count:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Collected {count} images for {name}")