# # import cv2 as cv
# # import numpy as np


# # face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# # age_net = cv.dnn.readNetFromCaffe("C:/Users/Gudniye Krishika/Desktop/opencv/age_deploy.prototxt", "C:/Users/Gudniye Krishika/Desktop/opencv/age_net.caffemodel")

# # # Corrected file paths (ensure correct directory name)
# # gender_net = cv.dnn.readNetFromCaffe(
# #     "C:/Users/Gudniye Krishika/Desktop/opencv/gender_deploy.prototxt",
# #     "C:/Users/Gudniye Krishika/Desktop/opencv/gender_net.caffemodel"
# # )

# # AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
# # GENDER_BUCKETS = ["Male", "Female"]

# # cap = cv.VideoCapture(0)

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     for (x, y, w, h) in faces:
# #         face = frame[y:y+h, x:x+w]
# #         if face.shape[0] == 0 or face.shape[1] == 0:
# #             continue
  
# #         face_blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)

# #         gender_net.setInput(face_blob)
# #         gender_preds = gender_net.forward()
# #         gender = GENDER_BUCKETS[gender_preds[0].argmax()]

# #         age_net.setInput(face_blob)
# #         age_preds = age_net.forward()
# #         age = AGE_BUCKETS[age_preds[0].argmax()]

# #         label = f"{gender}, {age}"
# #         cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #         cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# #     cv.imshow("Age & Gender Detection (Haar Cascades)", frame)

# #     if cv.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv.destroyAllWindows()




import cv2 as cv
import numpy as np

# Load Haar Cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load pre-trained age and gender detection models
age_net = cv.dnn.readNetFromCaffe(
    r"C:/Users/Gudniye Krishika/Desktop/opencv/age_deploy.prototxt", 
    r"C:/Users/Gudniye Krishika/Desktop/opencv/age_net.caffemodel"
)

gender_net = cv.dnn.readNetFromCaffe(
    r"C:/Users/Gudniye Krishika/Desktop/opencv/gender_deploy.prototxt",
    r"C:/Users/Gudniye Krishika/Desktop/opencv/gender_net.caffemodel"
)

# Age and Gender categories
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_BUCKETS = ["Male", "Female"]

# Start video capture
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv.namedWindow("Age & Gender Detection", cv.WINDOW_NORMAL)  # Ensures window opens properly

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue
  
        face_blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)

        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = GENDER_BUCKETS[gender_preds[0].argmax()]

        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = AGE_BUCKETS[age_preds[0].argmax()]

        label = f"{gender}, {age}" 
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.imshow("Age & Gender Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()




# import cv2 as cv
# import numpy as np
# import mediapipe as mp

# # Initialize Mediapipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# # Load Age & Gender Models (Make sure you have downloaded them)
# # age_net = cv.dnn.readNetFromCaffe(
# #     "age_deploy.prototxt", 
# #     "age_net.caffemodel"
# # )

# gender_net = cv.dnn.readNetFromCaffe(
#     "gender_deploy.prototxt", 
#     "gender_net.caffemodel"
# )
# age_net = cv.dnn.readNetFromCaffe(
#     r"C:/Users/Gudniye Krishika/Desktop/opencv/age_deploy.prototxt", 
#     r"C:/Users/Gudniye Krishika/Desktop/opencv/age_net.caffemodel"
# )

# gender_net = cv.dnn.readNetFromCaffe(
#     r"C:/Users/Gudniye Krishika/Desktop/opencv/gender_deploy.prototxt",
#     r"C:/Users/Gudniye Krishika/Desktop/opencv/gender_net.caffemodel"
# )


# # Labels for Age & Gender
# AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
# GENDER_BUCKETS = ["Male", "Female"]

# # Open Webcam
# cap = cv.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Webcam not found!")
#     exit()

# cv.namedWindow("Age & Gender Detection", cv.WINDOW_NORMAL)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to RGB for Mediapipe
#     rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     results = face_detection.process(rgb_frame)

#     if results.detections:
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             h, w, _ = frame.shape
#             x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

#             # Extract face ROI
#             face = frame[y:y+h, x:x+w]
#             if face.shape[0] == 0 or face.shape[1] == 0:
#                 continue

#             # Preprocess face for model
#             face_blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)

#             # Predict Gender
#             gender_net.setInput(face_blob)
#             gender_preds = gender_net.forward()
#             gender = GENDER_BUCKETS[gender_preds[0].argmax()]

#             # Predict Age
#             age_net.setInput(face_blob)
#             age_preds = age_net.forward()
#             age = AGE_BUCKETS[age_preds[0].argmax()]

#             # Draw results on frame
#             label = f"{gender}, {age}"
#             cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Show the frame
#     cv.imshow("Age & Gender Detection", frame)

#     # Exit when 'q' is pressed
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()
