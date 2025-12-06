import glob
import random
import cv2
import matplotlib.pyplot as plt
import modal
import os
from ultralytics import YOLO
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .uv_pip_install(
        ["ultralytics~=8.2.68", "opencv-python~=4.10.0", "term-image==0.7.1"]
    )
)

volume = modal.Volume.from_name("yolo-finetune", create_if_missing=True)
app = modal.App(name="blood_cell_localization", image=image)


@app.function(gpu="T4", volumes={"/data": volume}, timeout=3600)
def train_and_validate():
    volume.reload()
    model = YOLO("yolov10m.pt")

    results = model.train(
        data="/data/config.yaml",
        epochs=50,
        imgsz=640,
        project="/data/runs",
        name="yolov10m_run",
        plots=True,
    )

    volume.commit()
    metrics = model.val()

    return {
        "map50": metrics.box.map50,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "map50_95": metrics.box.map,
    }


@app.local_entrypoint()
def main():
    with volume.batch_upload() as batch:
        batch.put_file("config.yaml", "/data/config.yaml")
        batch.put_directory("data", "/data")

    metrics = train_and_validate.remote()

    print("Training Results: ")
    print(f"mAP@50: {metrics['map50']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"mAP@50-95: {metrics['map50_95']:.4f}")

    # Download runs folder
    print("Downloading runs from volume...")
    local_runs_dir = "runs"
    os.makedirs(local_runs_dir, exist_ok=True)
    
    for entry in volume.iterdir("/runs", recursive=True):
        if entry.type == modal.volume.FileEntryType.FILE:
            local_path = os.path.join(local_runs_dir, entry.path.replace("/runs/", "", 1))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in volume.read_file(entry.path):
                    f.write(chunk)
    print("Download complete.")
