# Importing required modules
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import mediapipe as mp
import time
# Load the pre-trained expression prediction model from the drive
def initialize_model():
    # Open and read the model file
    with open('mp_rf_model.p', 'rb') as f:
        model_data = pickle.load(f)
    # Return the loaded model
    return model_data['model']

# Set up the MediaPipe FaceMesh model with static_image_mode set to False for real-time video
def initialize_mediapipe_facemesh():
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Predict the expression using a frame and overlay landmarks on the frame
def predict_expression_from_frame(frame, clf, facemesh, label_classes):
    # Convert the frame color space to RGB and process it with FaceMesh
    results = facemesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Check if any faces are detected in the frame
    if not results.multi_face_landmarks:
        return frame, None, None

    # List to store the extracted landmarks
    landmark_data = []
    for landmarks in results.multi_face_landmarks:
        for landmark in landmarks.landmark:
            # Calculate the landmark's x and y coordinates on the frame
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            # Draw the landmark as a green circle on the frame
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            # Append the x, y, and z values of the landmark to the list
            landmark_data.extend([landmark.x, landmark.y, landmark.z])

    # Use the trained model to predict the expression using the landmarks
    prediction_index = clf.predict([landmark_data])[0]
    prediction_label = label_classes[prediction_index]
    confidence = np.max(clf.predict_proba([landmark_data]))
    prediction_expression = prediction_label.capitalize()

    return frame, prediction_expression, confidence

# Function to run live video prediction from the webcam
def live_video_prediction(clf, facemesh, label_classes):
    # Start video capture from the default webcam (0)
    cap = cv2.VideoCapture(0)
    
    # Loop until 'q' is pressed to quit
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Predict the expression using the current frame
        start_time = time.time()
        frame_with_landmarks, prediction_expression, confidence = predict_expression_from_frame(frame, clf, facemesh, label_classes)
        finish_time = time.time() - start_time

        print(finish_time*1000,"ms")
        # Display prediction on the frame if available
        if prediction_expression:
            cv2.putText(frame_with_landmarks, f"{prediction_expression} ({confidence*100:.2f}%)", 
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame with landmarks and predictions
        cv2.imshow('Facial Expression Recognition', frame_with_landmarks)

        # Press 'q' to exit the webcam window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()



def main():
    # Initialize the expression prediction model and the MediaPipe FaceMesh model
    clf = initialize_model()
    facemesh = initialize_mediapipe_facemesh()
    label_classes = ['very_low', 'low', 'high', 'very_high']

    # Run the live video prediction
    live_video_prediction(clf, facemesh, label_classes)


if __name__ == '__main__':
    main()