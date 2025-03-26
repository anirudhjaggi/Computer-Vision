import cv2
import mediapipe as mp
import time

# Load video file
cap = cv2.VideoCapture('Pose_Videos/1.mp4')

# Initialize MediaPipe Pose model
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Initialize drawing utilities
mpDraw = mp.solutions.drawing_utils

pTime = 0  # Previous time for FPS calculation

while True:
    success, img = cap.read()  # Read a frame from the video
    if not success:
        break  # Exit if video ends

    img = cv2.resize(img, (1200, 700))  # Resize the frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    results = pose.process(imgRGB)  # Process the frame for pose estimation

    if results.pose_landmarks:
        # Loop through detected landmarks and highlight specific key points
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape  # Get image dimensions
            cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values

            # Highlight specific landmarks (head, hands, knees)
            if id in [0, 15, 16, 27, 28]:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # Draw the pose skeleton
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 5)

    cv2.imshow("Image", img)  # Show the processed frame
    cv2.waitKey(1)  # Wait briefly to display the frame
