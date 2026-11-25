import time
import cv2
import mediapipe as mp

# MediaPipe Tasks API aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store raw detection output
current_landmarks = None


def print_result(result, output_image, timestamp_ms):
    """Callback triggered whenever MediaPipe finishes processing a frame."""
    global current_landmarks
    if result.hand_landmarks:
        current_landmarks = result.hand_landmarks[0]
    else:
        current_landmarks = None


# Configure hand landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    result_callback=print_result,
)

# Hand joint connections (21 points)
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
]

# Open default camera
cap = cv2.VideoCapture(0)
print("Starting webcam with MediaPipe Tasks... Press 'q' to exit.")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Format image for MediaPipe Tasks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Send frame asynchronously
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Safely copy active landmarks to avoid NoneType race conditions
        landmarks = current_landmarks

        if landmarks is not None:
            # Draw lines
            for connection in HAND_CONNECTIONS:
                p1 = landmarks[connection[0]]
                p2 = landmarks[connection[1]]
                cv2.line(
                    frame,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    (0, 255, 0),
                    2,
                )

            # Draw dots
            for landmark in landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                "Hand Tracked!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Show webcam window
        cv2.imshow("ASL Hand Tracking Test", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()