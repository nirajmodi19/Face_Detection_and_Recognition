# Importing the libraries
import cv2

# Importing the image
img = cv2.imread('test.jpg')

# Loading the Cascade files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_extended.xml ')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
recog = cv2.face.createLBPHFaceRecognizer()
recog.load('trained.yml')

# Defining the function used for detection
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.15, 10)
    print (faces)
    user = ['','Niraj', 'Manoj', 'Biplab']
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]    
        faceID = recog.predict(roi_gray)
        username = user[faceID]
        cv2.putText(frame, username, (x, y-10), cv2.FONT_ITALIC , 1, (0, 255, 0), 2)
    return frame

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = detect(gray, img)
 #Displaying the image
cv2.imshow('Show',cv2.resize(image, (400, 500)) )
cv2.waitKey(0)
cv2.destroyAllWindows()