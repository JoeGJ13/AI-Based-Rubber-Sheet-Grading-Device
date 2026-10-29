from ultralytics import YOLO

# 1. Load your best trained weights
weights_path = r"<best.pt path>"
model = YOLO(weights_path)

# 2. Run validation specifically on the 'test' split
metrics = model.val(
    data=r"D:\RSS Grader\Dataset",
    split="test",
    device="cpu"
)

# 3. Print test accuracy
print(f"\n--- Final Test Results ---")
print(f"Test Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
