from ultralytics import YOLO
import cv2
from twilio.rest import Client

# Load the trained YOLOv8 model
model = YOLO(r'C:\Users\Lenovo\Desktop\yolov8n.pt')

# Twilio config (replace with your credentials)

account_sid = 'xxxxxxxxxxxxxx'#twilio accountid
auth_token = 'xxxxxxxxxxxxxxxxxxxxxxxx'# twilio auth token
from_number = '+18543002785'  # Twilio number
to_number = '+xxxxxxxxxxx'  # Your phone number

client = Client(account_sid, auth_token)

# Start webcam
cap = cv2.VideoCapture(0)
weapon_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]

            # Assuming your weapon classes are named 'gun', 'knife', etc.
            if label.lower() in ['gun', 'knife']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Send SMS only once
                if not weapon_detected:
                    message = client.messages.create(
                        body=f'Alert: {label} detected!',
                        from_=from_number,
                        to=to_number
                    )
                    print("SMS Sent:", message.sid)
                    weapon_detected = True
            else:
                # Draw green box for other objects
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Weapon Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
