import cv2
#import numpy as np
#from py_mod.faceDetect import detect

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
user = ['User Unknown','Niraj Modi', 'Manoj Sharma', 'Biplab Kumar']

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('trained.yml')
cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, flags = cv2.CASCADE_SCALE_IMAGE)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        face = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face, 1.3, 5, flags = cv2.CASCADE_SCALE_IMAGE)
        if (len(eyes) == 2):
            if face is None:
                continue
            faceid = recognizer.predict(face)
            username = user[faceid]
            cv2.putText(img, username, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

