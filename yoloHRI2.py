from ultralytics import YOLO 
import cv2
import torch

print("Start")

'''
    COMPUTER VISION PART
'''
model = YOLO('yolov8m.pt')
image_path = '/Users/miguelmarombal/Desktop/HRI/FinalProject/yoloCode/MIDCHECK.jpeg'
image = cv2.imread(image_path)

# Dimensions of the image
height, width, _ = image.shape

# Split the image into three sections
left_section = image[:, :width // 3]
middle_section = image[:, width // 3: 2 * width // 3]
right_section = image[:, 2 * width // 3:]

sections = [left_section, middle_section, right_section]
section_names = ["Left", "Middle", "Right"]

# Analyze each section
for i, section in enumerate(sections):
    results = model(section)
    result = results[0]
    
    # Check if 'person' is detected in this section
    person_detected = any(result.names[box.cls[0].item()] == 'person' for box in result.boxes)

    print(f"{section_names[i]} Section - Person Detected: {'Yes' if person_detected else 'No'}")

    # Optionally, display each section with annotations
    annotated_section = result.plot()
    cv2.imshow(f"{section_names[i]} Section", annotated_section)

cv2.waitKey(0)
cv2.destroyAllWindows()
