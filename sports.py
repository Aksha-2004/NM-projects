import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Input: Video Footage ===
video_path = 'runner.mp4'  # Or use 0 for live webcam
cap = cv2.VideoCapture(video_path)

# === Output Data ===
trajectory_points = []

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))  # Resize for consistency
    mask = fgbg.apply(frame)

    # Find contours of moving objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 300:  # Filter out small noise
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        trajectory_points.append((cx, cy))

        # Draw player bounding box and center
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 2, (0,0,255), -1)

    # Draw trajectory
    for i in range(1, len(trajectory_points)):
        cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (255, 0, 0), 1)

    cv2.imshow("Player Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# === Generate Heatmap ===
if trajectory_points:
    x_vals, y_vals = zip(*trajectory_points)
    heatmap, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=(64, 36), range=[[0, 640], [0, 360]])

    plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest')
    plt.title("Player Heatmap")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.colorbar()
    plt.show()