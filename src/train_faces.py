# train_faces.py
import cv2
import os
import numpy as np
import pickle

BASE_DIR = "datasets"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace("_", " ").strip()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# Save label IDs
os.makedirs("data", exist_ok=True)
with open("data/labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("data/face_trainer.yml")

print(f"[INFO] Training complete. Trained on {len(label_ids)} faces.")
