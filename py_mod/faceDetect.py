import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if (len(faces)==0):
        return None
    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]
    return gray[y:y+h, x:x+w]

#cam = cv2.VideoCapture(0)
#
#while True:
#    _, frame = cam.read()
#    _, img, _ = detect(frame)
#    cv2.imshow('canvas', img)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cam.release()
#cv2.destroyAllWindows()    

    
    


        

