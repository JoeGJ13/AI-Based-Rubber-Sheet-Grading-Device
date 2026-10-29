import os
from ultralytics import YOLO

def main():
    # Direct path to your dataset folder
    dataset_path = r"<Dataset folder Path>"

    # Load pretrained YOLO classification model
    model = YOLO("yolo11n-cls.pt")

    # Start training
    model.train(
        data=dataset_path,
        epochs=80,
        imgsz=448,
        batch=16,
        patience=20,
        device="cpu",       # Using CPU on your Ryzen 5 9600X
        fliplr=0.5,
        flipud=0.5,
        project="rubber_run",
        name="rss4_vs_rss5"
    )

if __name__ == "__main__":
    main()
