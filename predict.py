import os
from ultralytics import YOLO

# Path to your newly trained model
weights_path = r"<best.pt path>"
model = YOLO(weights_path)

# Pick any test image from your dataset (or a completely new photo)
test_image = r"<image path>"

results = model(test_image)

probs = results[0].probs
top_idx = probs.top1
grade_name = results[0].names[top_idx]
confidence = probs.top1conf.item() * 100

print(f"\n==============================")
print(f" Classified Grade : {grade_name}")
print(f" Confidence       : {confidence:.2f}%")
print(f"==============================\n")
