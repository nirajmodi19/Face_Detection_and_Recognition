import cv2

face_cascade = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')

def detect(frame):
    gray = cv2. cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)        
    return gray[y:y+h, x:x+w], frame, faces[0]

cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    _, img, _ = detect(frame)
    cv2.imshow('canvas', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()    

    
    


        

