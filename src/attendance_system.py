# attendance_system.py
import cv2
import pickle
from utils import mark_attendance

# Load trained model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("data/face_trainer.yml")

with open("data/labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("[INFO] Starting attendance system. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi)

        if confidence < 70:  # lower = better
            name = labels[id_]
            mark_attendance(name)
            color = (0, 255, 0)
            label = f"{name} ({round(confidence, 2)})"
        else:
            color = (0, 0, 255)
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
