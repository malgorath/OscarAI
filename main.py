import cv2
import numpy as np
import tensorflow as tf
import os


os.environ['KERAS_BACKEND'] = 'tensorflow'

# TODO: Need to find a model for this to work.
# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a list of emotions
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    
    # Read the video frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each face detected
    for (x, y, w, h) in faces:
        
        # Extract the face
        face = gray[y:y+h, x:x+w]
        
        # Resize the face to match the input size of the model
        resized = cv2.resize(face, (48, 48))
        
        # Reshape the input to match the input shape of the model
        input_data = np.array(resized).reshape(-1, 48, 48, 1).astype(np.float32)
        
        # Normalize the input data
        input_data = input_data / 255.0
        
        # Make a prediction on the input data
        prediction = model.predict(input_data)[0]
        
        # Get the index of the predicted emotion
        index = np.argmax(prediction)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the predicted emotion
        cv2.putText(frame, EMOTIONS[index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotional Detection', frame)

    # If 'q' is pressed, break the loop
    if cv3.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()

