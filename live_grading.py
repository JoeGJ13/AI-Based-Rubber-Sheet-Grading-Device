import cv2
import time
from ultralytics import YOLO

# 1. Path to your best trained model weights
weights_path = r"<best.pt path>"
model = YOLO(weights_path)

# 2. Phone IP camera URL (verify this matches your phone app's address)
IP_CAMERA_URL = "http://<IP camera id>/video"

print(f"Connecting to phone camera at {IP_CAMERA_URL}...")
cap = cv2.VideoCapture(IP_CAMERA_URL)

if not cap.isOpened():
    print("Error: Could not connect to camera stream. Check the IP and Wi-Fi connection.")
    exit()

print("Connected! Press 'q' on the video window to exit.")

# Confidence parameters
# Requires the top prediction to be at least 88% sure before declaring a grade
CONFIDENCE_THRESHOLD = 88.0

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Stream ended or disconnected.")
        break

    # Run YOLO classification
    results = model(frame, imgsz=448, verbose=False)

    # Extract probability values
    probs = results[0].probs
    top_idx = probs.top1
    top_conf = probs.top1conf.item() * 100
    detected_class = results[0].names[top_idx]

    # Calculate FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Apply confidence threshold
    if top_conf < CONFIDENCE_THRESHOLD:
        status_text = f"No Rubber Sheet Detected ({top_conf:.1f}%)"
        banner_color = (80, 80, 80)      # Gray banner
        text_color = (200, 200, 200)      # Dim white text
    else:
        status_text = f"Grade: {detected_class} ({top_conf:.1f}%)"
        # Green for RSS4, Amber/Orange for RSS5
        banner_color = (0, 100, 0) if detected_class == "RSS4" else (0, 70, 180)
        text_color = (0, 255, 0) if detected_class == "RSS4" else (0, 165, 255)

    # Draw top banner box
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 75), banner_color, -1)

    # Put grade/status text
    cv2.putText(frame, status_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

    # Put FPS in the corner
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    # Show live stream
    cv2.imshow("Rubber Sheet Inspector", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
