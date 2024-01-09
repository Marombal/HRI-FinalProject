from ultralytics import YOLO 
import cv2
import torch

print("Start")

'''
    COMPUTER VISION PART
'''
model = YOLO('yolov8m.pt')
vid = cv2.VideoCapture(0)  # Use 0 for the default webcam

def draw_boxes(img, boxes, classes, confidences):
    for box, cls, conf in zip(boxes, classes, confidences):
        if conf > 0.75:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{cls} {conf:.2f}"
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

while vid.isOpened():
    success, frame = vid.read()
    if not success:
        break

    # Dimensions of the frame
    height, width, _ = frame.shape

    # Split the frame into three sections
    left_section = frame[:, :width // 3]
    middle_section = frame[:, width // 3: 2 * width // 3]
    right_section = frame[:, 2 * width // 3:]

    sections = [left_section, middle_section, right_section]
    section_names = ["Left", "Middle", "Right"]

    for i, section in enumerate(sections):
        results = model(section)
        result = results[0]

        # Extract boxes, class names, and confidences
        boxes = [box.xyxy[0].cpu().numpy() for box in result.boxes]
        class_names = [result.names[box.cls[0].item()] for box in result.boxes]
        confidences = [box.conf[0].item() for box in result.boxes]

        # Draw boxes with confidence > 75%
        draw_boxes(section, boxes, class_names, confidences)

        # Check if 'person' is detected in this section with confidence > 75%
        person_detected = any(cls == 'person' and conf > 0.75 for cls, conf in zip(class_names, confidences))
        print(f"{section_names[i]} Section - Person Detected: {'Yes' if person_detected else 'No'}")

    # Display the full frame with annotations
    cv2.imshow("Webcam - YOLOv8 Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
