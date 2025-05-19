import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

# Background subtractor for smoke detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([18, 50, 50])
    upper_fire = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    fire_pixels = cv2.countNonZero(mask)
    return fire_pixels > 5000

def detect_smoke(frame):
    fg_mask = bg_subtractor.apply(frame)
    blurred = cv2.GaussianBlur(fg_mask, (21, 21), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    motion_pixels = cv2.countNonZero(thresh)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smoke_area = cv2.inRange(gray, 100, 180)
    smoke_pixels = cv2.countNonZero(smoke_area)

    return motion_pixels > 3000 and smoke_pixels > 5000

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame = cv2.resize(frame, (640, 480))

    fire = detect_fire(frame)
    smoke = detect_smoke(frame)

    # Determine message and color
    if fire and smoke:
        message = "🔥💨 Fire & Smoke Detected!"
        color = (0, 0, 255)
    elif fire:
        message = "🔥 Fire Detected!"
        color = (0, 0, 255)
    elif smoke:
        message = "💨 Smoke Detected!"
        color = (128, 128, 128)
    else:
        message = "✅ No Threat Detected"
        color = (0, 255, 0)

    # Print the message in terminal
    print(message)

    # Display message on screen
    cv2.putText(frame, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 3)

    # Show video output
    cv2.imshow("Fire and Smoke Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()