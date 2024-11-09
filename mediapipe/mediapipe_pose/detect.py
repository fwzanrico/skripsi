import mediapipe as mp
import cv2
import numpy as np
import pickle

class FaceMeshDetector:
    def __init__(self):
        # Initialize mediapipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(128,0,128), thickness=2, circle_radius=1)
        
        # Load the pre-trained model
        with open('./lm_pose.p', 'rb') as model_file:
            rf_classifier = pickle.load(model_file)
            self.clf = rf_classifier['model']
            
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
            return self.analyze_face(image, results.multi_face_landmarks[0])
        return image, None, None

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

def main():
    detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image, bbox, prediction = detector.process_frame(frame)
        
        cv2.imshow('Face Mesh Detection', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()