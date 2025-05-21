import cv2
import numpy as np

# --- 1. Load Face Detector ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Load Emotion Recognition Model ---
emotion_model = cv2.dnn.readNetFromONNX("emotion-ferplus-12-int8.onnx")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']  # Update as needed

# --- Capture Video ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi_normalized = roi_gray_resized / 255.0

        # Add channel dimension: shape (48, 48) -> (48, 48, 1)
        roi_expanded = np.expand_dims(roi_normalized, axis=2)
        # Now create blob: shape (1, 1, 48, 48)
        image_blob = cv2.dnn.blobFromImage(roi_expanded, scalefactor=1.0, size=(48, 48), mean=0, swapRB=False, crop=False)

        emotion_model.setInput(image_blob)
        preds = emotion_model.forward()
        emotion_probability = np.max(preds)
        label_index = np.argmax(preds)
        predicted_emotion = emotion_labels[label_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Emotion Recognition', frame)

    # Exit on pressing 'q' or when the window is closed
    if cv2.getWindowProperty('Emotion Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()