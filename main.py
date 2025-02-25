import cv2
import numpy as np
import face_recognition
import os
import pyttsx3
from datetime import datetime


# Path where face images are stored
IMAGE_PATH = 'images'  # Change this path to your image folder

# Load images and extract names
images = []
class_roll = []
image_list = os.listdir(IMAGE_PATH)
for img_roll in image_list:
    img = cv2.imread(f'{IMAGE_PATH}/{img_roll}')
    images.append(img)
    class_roll.append(os.path.splitext(img_roll)[0])  # Remove file extension

# Function to encode known images
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encode_list.append(encodings[0])
    return encode_list

def speak(text):
     engine = pyttsx3.init()
     engine.say(text)
     engine.runAndWait()
     
marked_roll =set()
# Function to mark attendance
marked_rolls = set()

def mark_attendance(roll):
    if roll not in marked_rolls:
        marked_rolls.add(roll)
        with open('attendance.csv', 'a') as f:
            now = datetime.now().strftime('%Y-%m-%d,%H:%M:%S')
            f.write(f'Roll Number: {roll},{now}\n')
        speak(f"Attendance marked for roll number {roll}")





# Encode known faces
known_encodings = find_encodings(images)
print("Encoding complete!")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    faces_in_frame = face_recognition.face_locations(small_frame)
    encodings_in_frame = face_recognition.face_encodings(small_frame, faces_in_frame)
    
    for face_encoding, face_location in zip(encodings_in_frame, faces_in_frame):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        match_index = np.argmin(face_distances)
        
        if matches[match_index]:
            roll = class_roll[match_index]
            mark_attendance(roll)
            
            y1, x2, y2, x1 = [i * 4 for i in face_location]  # Scale back
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, roll, (x1 + 6, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

