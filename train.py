import cv2
import os
import numpy as np
from py_mod.faceDetect import detect
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    
    for dir_name in dirs:
        
        label = int(dir_name.replace("s", ""))
        user_dir_path = data_folder_path + '/' + dir_name
        
        user_images_name = os.listdir(user_dir_path)
        
        for user_image in user_images_name:
            
            if user_image.startswith("."):
                continue
            
            image_path = user_dir_path + '/' + user_image
            
            image = cv2.imread(image_path)
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = detect(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
        
    return faces, labels

faces, labels = prepare_training_data('training-data')

face_recognizer = cv2.face.createLBPHFaceRecognizer()

face_recognizer.train(faces, np.array(labels))
face_recognizer.save("trained.yml")
cv2.destroyAllWindows()
            
            
    
    
    