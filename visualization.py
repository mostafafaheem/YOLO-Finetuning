import glob
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import sys

# Run using:
# python -m visualization 8
# OR
# python -m visualization 10

testing_images = glob.glob("data_v2/images/test/*.jpg")
random_samples = random.sample(testing_images, 3)
plt.figure(figsize=(10, 5))

assert len(sys.argv) > 0 and len(sys.argv) < 3, "Invalid number of arguments!"
assert sys.argv[1] == "8" or sys.argv[1] == "10", "Model unavailable!"
weights_path = ""

if sys.argv[1] == "8":
    weights_path = "runs/yolov8l_run/weights/best.pt"
else:
    weights_path = "runs/yolov10m_run/weights/best.pt"

model = YOLO(weights_path)

import os
import cv2

# Create a folder to store the results
output_dir = "inference_results"
os.makedirs(output_dir, exist_ok=True)

for i, image_path in enumerate(random_samples):
    prediction = model.predict(image_path, conf=0.25)
    prediction_img = prediction[0].plot()
    save_path = os.path.join(output_dir, f"result_{i}.jpg")
    cv2.imwrite(save_path, prediction_img)
    print(f"Saved {save_path}")
