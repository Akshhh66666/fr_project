# capture_faces.py
import cv2
import os

name = input("Enter student name (use underscores for spaces): ")
path = f"dataset/{name}"

if not os.path.exists(path):
    os.makedirs(path)

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("\n[INFO] Capturing face samples. Look at the camera and wait...")
count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"{path}/{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f"Samples: {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Capturing Faces', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:  # ESC to exit early
        break
    elif count >= 30:  # capture 30 samples
        break

print("\n[INFO] Done capturing faces.")
cam.release()
cv2.destroyAllWindows()

