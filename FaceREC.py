from pickle import FRAME
import face_recognition
import cv2
import numpy as np
import csv
import os
import glob
from datetime import datetime

video_capture = cv2.VideoCapture(0)

sallu_bhai = face_recognition.load_image_file("C:/Users/acer/Documents/FaceRecg/FaceRecg/Photos/SalluBhai.jpg")
sallu_encoding = face_recognition.face_encoding(sallu_bhai)[0]

rashmika = face_recognition.load_image_file("C:/Users/acer/Documents/FaceRecg/FaceRecg/Photos/Rashmika.jpg")
rashmika_encoding = face_recognition.face_encoding(rashmika)[0]

Cristianoo = face_recognition.load_image_file("C:/Users/acer/Documents/FaceRecg/FaceRecg/Photos/CR7.jpg")
CR7_encoding = face_recognition.face_encoding(Cristianoo)[0]

know_face_encoding = [sallu_encoding, rashmika_encoding, CR7_encoding]
know_face_name = ["Salman Khan", "Rashmika Mandhana", "Cristiano Ronaldo"]

studs = know_face_name.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+".csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(know_face_encoding, face_encoding)
            name  = ""
            face_distance = face_recognition.face_distance(know_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = know_face_name[best_match_index]
            face_names.append(name)
            
            if name in know_face_name:
                if name in studs:
                    studs.remove(name)
                    print(studs)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("FaceDetection System", frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    

video_capture.release()
cv2.destroyAllWindows()
f.close()
