Face Recognition Attendance System:

This project is a "Face Recognition-based Attendance System"
that detects faces using OpenCV and marks attendance automatically.

How to use:
- First create an images folder in the same directory as your main.py file.
- Then save your images of the attendants as (roll no of the attendant).jpg in the images folder.
- Now run the file and the program will open your webcam and detect the face.
- Now it will print the roll number, date and time of the attendant.

📌 Features:
- Detects and recognizes faces in real-time.
- Marks attendance and saves it in 'attendance.csv'.
- Uses "Text-to-Speech (TTS)" to confirm attendance.

🛠 Requirements
Install dependencies using:
pip install opencv-python numpy face-recognition pyttsx3
