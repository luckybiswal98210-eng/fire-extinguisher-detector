from ultralytics import YOLO
import cv2
import time
import pyttsx3

# Load your trained YOLO model
model = YOLO("C:/Users/risha/Downloads/best.pt")

# Fix class names (since your model shows {0: '0'})
model.names = {0: "fire extinguisher"}

# Text-to-speech engine
engine = pyttsx3.init()
last_spoken = 0

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]  # Should now say "fire extinguisher"

            # Draw red box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Estimate distance (fake formula — replace if you calibrate)
            distance = round(1000 / (x2 - x1), 2)  # meters approx
            cv2.putText(frame, f"{distance} m", (x2 - 100, y2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Speak every 3 seconds
            if time.time() - last_spoken > 3:
                engine.say(label)
                engine.runAndWait()
                last_spoken = time.time()

    cv2.imshow("Live Fire Extinguisher Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
