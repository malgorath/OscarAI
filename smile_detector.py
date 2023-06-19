import cv2
import numpy as np

# Load the face and emotion classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

# Set up the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face and convert it to grayscale for emotion detection
        roi_gray = gray[y:y + h, x:x + w]

        # Detect the smile in the face
        smiles = emotion_classifier.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=15, minSize=(25, 25))

        # Draw rectangles around the detected smiles
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the program when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()