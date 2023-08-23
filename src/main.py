import cv2  # Importing OpenCV
from imutils import face_utils  # Importing face_utils from imutils
import imutils
import dlib  # Importing dlib
import numpy as np  # Importing numpy as np
from pygame import mixer  # Importing mixer from pygame
import time

# Starting live video capture form webcam
cap = cv2.VideoCapture(0)


# Below, we have created a function which
# will give us the magnitude of the length
# of the vector between any two of the six
# landmarks of an eye
def eye_landmark_vector_norm(ld1, ld2):
    return np.linalg.norm(ld1 - ld2)


# Now, we have used the above function and created another function to compute the aspect ratio of an individual
# eye using the formula for finding the aspect ratio of an eye
def get_ear(eye):
    # Finding the vector norm between all the pairs of landmarks
    a = eye_landmark_vector_norm(eye[1], eye[5])
    b = eye_landmark_vector_norm(eye[2], eye[4])
    c = eye_landmark_vector_norm(eye[0], eye[3])

    # Computing the eye aspect ratio using the formula
    ear = (a + b) / (2.0 * c)

    # Finally, returning the eye aspect ratio
    return ear


def label(image, text, x, y, w, h, color):
    cv2.putText(image, text, (x + 20, y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
    cv2.line(image, (x, y), (x + int(w / 4), y), color, 2)
    cv2.line(image, (x, y), (x, y + int(h / 4)), color, 2)
    cv2.line(image, (x + int(w / 1.5), y), (x + w, y), color, 2)
    cv2.line(image, (x + w, y), (x + w, y + int(h / 4)), color, 2)
    cv2.line(image, (x, y + int(h / 1.5)), (x, y + h), color, 2)
    cv2.line(image, (x, y + h), (x + int(w / 4), y + h), color, 2)
    cv2.line(image, (x + int(w / 1.5), y + h), (x + w, y + h), color, 2)
    cv2.line(image, (x + w, y + int(h * 3 / 4)), (x + w, y + h), color, 2)


mixer.init()
mixer.music.set_volume(2.0)
mixer.music.load("1692545222372asxvhq9-voicemaker.in-speech.mp3")
# Next, we load the face detection and landmark predictor pre-trained models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ld_prediction_model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Setting the thresholds for the eye aspect ratio and the number of consecutive frames for which
# the eyes can remain closed
ear_threshold = 0.3
consecutive_frames_threshold = 16

# Initialising the counter variable for counting the number of consecutive frames
counter = 0

# Now, let's go ahead and extract the landmarks of each eye with the following statements
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:

    ret, frame = cap.read()  # Reading from the frame
    frame = imutils.resize(frame, width=450)  # Resizing the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Creating a gray frame

    # Detecting faces from the haarcascade face detection model
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                          minNeighbors=9, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        face = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = ld_prediction_model(gray_frame, face)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]
        left_ear = get_ear(left_eye)
        right_ear = get_ear(right_eye)

        ear = (left_ear + right_ear) / 2.0

        pts_left_eye = np.array(left_eye, np.int32)
        pts_left_eye = pts_left_eye.reshape((-1, 1, 2))

        pts_right_eye = np.array(right_eye, np.int32)
        pts_right_eye = pts_right_eye.reshape((-1, 1, 2))

        cv2.polylines(frame, [pts_left_eye], True, (255, 255, 255), 1)
        cv2.polylines(frame, [pts_right_eye], True, (255, 255, 255), 1)

        if ear < ear_threshold:
            counter += 1
            print(counter)
            if counter == consecutive_frames_threshold:
                mixer.music.play()
                label(frame, "Driver Drowsy!", x, y, w, h, (255, 255, 255))
                cv2.imwrite("drowsy.jpg", frame)

        else:
            counter = 0
    if ret:
        cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
