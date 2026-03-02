import cv2
import numpy as np

# Load pre-trained models
age_net = cv2.dnn.readNetFromCaffe("models/deploy_age.prototxt", "models/age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("models/deploy_gender.prototxt", "models/gender_net.caffemodel")

# Define age and gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male','Female']

# Start webcam video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for age and gender detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), (104.0, 177.0, 123.0))
    
    # Detect gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    # Detect age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    # Prepare the display text
    text = f"{gender}, Age: {age}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Age and Gender Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
