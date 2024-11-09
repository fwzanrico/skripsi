import mediapipe as mp
import cv2
import numpy as np
import pickle
mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128,0,128),thickness=2,circle_radius=1)


# Load pre-trained model and scaler
with open('./lm_pose.p', 'rb') as model_file:
    rf_classifier = pickle.load(model_file)
    clf = rf_classifier['model']
# camera stream:

cap = cv2.VideoCapture(0)  # chose camera index (try 1, 2, 3)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # number of faces to track in each frame
        refine_landmarks=True,  # includes iris landmarks in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:  # no frame input
            print("Ignoring empty camera frame.")
            continue
        # To improve performance, optionally mark the image as not writeable to

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV
        img_h , img_w, img_c = image.shape
        face_2d = []
        face_3d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                        if idx ==1:
                            nose_2d = (lm.x * img_w,lm.y * img_h)
                            nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                        x,y = int(lm.x * img_w),int(lm.y * img_h)

                        face_2d.append([x,y])
                        face_3d.append(([x,y,lm.z]))
                coords = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                flattened_landmarks = np.array(coords).flatten()

                #Get 2d Coord
                face_2d = np.array(face_2d,dtype=np.float64)

                face_3d = np.array(face_3d,dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length,0,img_h/2],
                                    [0,focal_length,img_w/2],
                                    [0,0,1]])
                distortion_matrix = np.zeros((4,1),dtype=np.float64)

                success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


                #getting rotational of face
                rmat,jac = cv2.Rodrigues(rotation_vec)

                angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360



                nose_3d_projection,jacobian = cv2.projectPoints(nose_3d,rotation_vec,translation_vec,cam_matrix,distortion_matrix)
                ##print(nose_3d_projection, jacobian)
                
                p1 = (int(nose_2d[0]),int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] -x *10))

                cv2.line(image,p1,p2,(255,0,0),3)

                cv2.putText(image,"x: " + str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv2.putText(image,"y: "+ str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv2.putText(image,"z: "+ str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pose_val = np.array([x,y,z])

                feature_vector = np.concatenate([flattened_landmarks, pose_val])
                ##print(feature_vector)
                feature_vector = feature_vector.reshape(1, -1)
                prediction = clf.predict(feature_vector)
                predicted_class = prediction[0]

                # Display the prediction on the video frame
                cv2.putText(image, f"Prediction: {predicted_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
               
                ##gaze.gaze(image, results.multi_face_landmarks[0])  # gaze estimation
                mp_drawing.draw_landmarks(image=image,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=drawing_spec,
                                    connection_drawing_spec=drawing_spec)
        
        cv2.imshow('output window', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break
cap.release()
