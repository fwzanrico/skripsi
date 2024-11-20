import mediapipe as mp
import cv2
import numpy as np
import pickle
import time
from datetime import datetime
import csv
import os
import uuid

class FaceMeshDetector:
    def __init__(self,session_id):
        # Existing initialization code
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(128,0,128), thickness=2, circle_radius=1)
        
        # Load the pre-trained model
        with open('./lm_pose_data_kurasi.p', 'rb') as model_file:
            rf_classifier = pickle.load(model_file)
            self.clf = rf_classifier['model']
            
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize session-specific folders and logging
        self.initialize_session(session_id)
    
    def initialize_session(self, session_id):
        """Initialize session-specific folders and logging files."""
        # Create a unique session ID
        self.session_id = session_id
        
        # Create session folder structure
        self.session_folder = os.path.join('sessions', self.session_id)
        self.images_folder = os.path.join(self.session_folder, 'images')
        
        # Create necessary directories
        os.makedirs(self.session_folder, exist_ok=True)
        os.makedirs(self.images_folder, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(self.session_folder, f'detection_log_{self.session_id}.csv')
        self.initialize_logging()
    
    def initialize_logging(self):
        """Initialize the CSV log file with headers."""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 
                'Face_Detected', 
                'Class', 
                'Computational_Time',
                'Image_Path'
            ])
    
    def save_frame(self, frame, prediction):
        """
        Save the current frame to the images folder.
        
        Args:
            frame: The image frame to save
            prediction: The current prediction label
            
        Returns:
            str: The relative path to the saved image
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_filename = f'frame_{timestamp}_{prediction if prediction else "no_face"}.jpg'
        image_path = os.path.join(self.images_folder, image_filename)
        cv2.imwrite(image_path, frame)
        return os.path.join('images', image_filename)  # Return relative path
    
    def log_detection(self, face_detected, prediction, comp_time, image_path):
        """
        Log detection results to CSV file
        
        Args:
            face_detected (int): 1 if face detected, 0 if not
            prediction (str): Predicted class or None if no face detected
            comp_time (float): Computation time in seconds
            image_path (str): Relative path to the saved image
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                face_detected,
                prediction if prediction else 'None',
                f"{comp_time:.4f}",
                image_path
            ])

    def process_frame(self, frame):
        start_time = time.time()  # Start timing
        
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        comp_time = time.time() - start_time  # Calculate computation time
        
        if results.multi_face_landmarks:
            image, bbox, prediction = self.analyze_face(image, results.multi_face_landmarks[0])
            # Save the processed frame and log the detection
            image_path = self.save_frame(image, prediction)
            self.log_detection(1, prediction, comp_time, image_path)
            return image, bbox, prediction, comp_time
        
        # Save the frame and log even when no face is detected
        image_path = self.save_frame(image, None)
        self.log_detection(0, None, comp_time, image_path)
        return image, None, None, comp_time

    # ... [keep the analyze_face method unchanged] ...
    def analyze_face(self, image, face_landmarks):
        img_h, img_w, _ = image.shape
        face_2d = []
        face_3d = []
        
        # Get the face landmarks
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in [33, 263, 1, 61, 291, 199]:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        # Calculate pose
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        # Camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h/2],
            [0, focal_length, img_w/2],
            [0, 0, 1]
        ])
        distortion_matrix = np.zeros((4, 1), dtype=np.float64)
        


        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        # Get angles
        x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
        
        # Draw nose direction
        nose_3d_projection, _ = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)

        # Get features for prediction
        coords = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
        flattened_landmarks = np.array(coords).flatten()
        pose_val = np.array([x, y, z])
        feature_vector = np.concatenate([flattened_landmarks, pose_val]).reshape(1, -1)
        
        # Make prediction
        prediction = self.clf.predict(feature_vector)[0]
        
        # Draw bounding box
        x_coordinates = [landmark.x * img_w for landmark in face_landmarks.landmark]
        y_coordinates = [landmark.y * img_h for landmark in face_landmarks.landmark]
        
        x_min = int(min(x_coordinates))
        x_max = int(max(x_coordinates))
        y_min = int(min(y_coordinates))
        y_max = int(max(y_coordinates))
        
        # Add padding to the bounding box
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_w, x_max + padding)
        y_max = min(img_h, y_max + padding)
        
        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Add prediction text with background
        text = f"Prediction: {prediction}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, 
                    (x_min - 5, y_min - text_h - 15),
                    (x_min + text_w + 5, y_min - 5),
                    (0, 255, 0), -1)
        cv2.putText(image, text, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return image, (x_min, y_min, x_max, y_max), prediction
    # Rest of the FaceMeshDetector class remains the same...
    # (keeping all existing methods unchanged)

def main():
    session_name = input("enter session id with a format of subject_activity: ")
    detector = FaceMeshDetector(session_name)
    cap = cv2.VideoCapture(0)
    
    print(f"Session ID: {detector.session_id}")
    print(f"Saving data to: {detector.session_folder}")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image, bbox, prediction, comp_time = detector.process_frame(frame)
        
        cv2.imshow('Face Mesh Detection', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()