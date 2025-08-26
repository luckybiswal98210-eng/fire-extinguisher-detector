from ultralytics import YOLO
import cv2
import math
import pyttsx3  # for text-to-speech (it will speak the object name)

# Load the trained YOLO model (update path if needed)
model = YOLO(r"C:\Users\risha\Downloads\best.pt")  # change path if your best.pt is elsewhere

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    for r in results:
        boxes = r.boxes  # bounding boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # confidence
            cls = int(box.cls[0])  # class id
            label = model.names[cls]  # class name (like Fire Extinguisher)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Approximate distance (simple logic: larger box -> closer object)
            box_height = y2 - y1
            distance = round(5000 / box_height, 2)  # fake calculation for demo
            cv2.putText(frame, f"Dist: {distance} cm", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Speak the detected object name
            engine.say(label)
            engine.runAndWait()

            # (Extra) Crack detection placeholder
            # Here you can add a second YOLO model trained for cracks
            # For now, we just mark with red if confidence < 0.6
            if conf < 0.6:
                cv2.putText(frame, "⚠ Possible Crack/Defect!", (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show result
    cv2.imshow("Live Fire Extinguisher Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
