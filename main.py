import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)
print('Venkatesh')
print('Ramu')

# Constants
EYE_CLOSED_THRESHOLD = 0.2  # Adjust based on testing
WARNING_DURATION = 3  # Seconds
FRAME_COUNTER_THRESHOLD = 10  # Frames to trigger warning

# Eye landmarks (MediaPipe's 468 landmarks)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_landmarks, landmarks):
    # Extract (x, y) coordinates from NormalizedLandmark objects
    def get_coords(index):
        landmark = landmarks[eye_landmarks[index]]
        return np.array([landmark.x, landmark.y])

    # Compute Eye Aspect Ratio (EAR)
    vertical_dist1 = np.linalg.norm(get_coords(1) - get_coords(5))
    vertical_dist2 = np.linalg.norm(get_coords(2) - get_coords(4))
    horizontal_dist = np.linalg.norm(get_coords(0) - get_coords(3))
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    warning_counter = 0
    warning_active = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Calculate EAR for both eyes
            left_ear = calculate_ear(LEFT_EYE_LANDMARKS, landmarks)
            right_ear = calculate_ear(RIGHT_EYE_LANDMARKS, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0

            # Check if eyes are closed or looking away
            if avg_ear < EYE_CLOSED_THRESHOLD:
                warning_counter += 1
            else:
                warning_counter = max(0, warning_counter - 1)

            # Trigger warning
            if warning_counter > FRAME_COUNTER_THRESHOLD:
                cv2.putText(
                    frame,
                    "WARNING: Eyes moved away from screen!",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                warning_active = True
            else:
                warning_active = False

        else:
            # No face detected
            cv2.putText(
                frame,
                "WARNING: Face not detected!",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Online Proctoring", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()