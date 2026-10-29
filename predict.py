import os
from ultralytics import YOLO

# Path to your newly trained model
weights_path = r"D:\RSS Grader\Panisthalam\runs\classify\rubber_run\rss4_vs_rss5-2\weights\best.pt"
model = YOLO(weights_path)

# Pick any test image from your dataset (or a completely new photo)
test_image = r"D:\RSS Grader\Dataset\test\RSS4\IMG_20261019_112403580_jpg.rf.9cc8f3bdd2ddc669d48ce5c33203c091.jpg"

results = model(test_image)

probs = results[0].probs
top_idx = probs.top1
grade_name = results[0].names[top_idx]
confidence = probs.top1conf.item() * 100

print(f"\n==============================")
print(f" Classified Grade : {grade_name}")
print(f" Confidence       : {confidence:.2f}%")
print(f"==============================\n")
