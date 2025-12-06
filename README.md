# YOLO-Finetuning

This repository contains scripts and utilities to finetune YOLOv8l and YOLOv10m models for blood cell localization (White Blood Cells, Red Blood Cells, and Platelets). The training and validation pipelines are configured to run efficiently on cloud GPUs using `modal`.

## Project Structure

- **Dataset formatting:**
  - `data.csv`: Source dataset containing bounding box annotations (not necessarily included).
  - `reformatter.py`: Processes the dataset from CSV format into YOLO-compatible format (`data_v2/` directory structure with `.txt` label files) and splits it into train/test.
  - `create_config_file.py`: Generates the `config.yaml` file needed by YOLO for defining paths and class names.
  
- **Training (via Modal):**
  - `yolov8l.py`: Modal application script to train and validate a YOLOv8l model on a T4 GPU.
  - `yolov10m.py`: Modal application script to train and validate a YOLOv10m model.

- **Evaluation & Inference:**
  - `metrics.py`: Utility to parse the `results.csv` from training runs and extract key metrics (Precision, Recall, mAP).
  - `visualization.py`: Script to run inference on random testing images and save the bounding box predictions.

## Setup

1. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Authenticate with Modal (required for training scripts):
   ```bash
   python -m modal setup
   ```

## Usage Workflow

> [!NOTE] 
> If you already have your dataset properly structured and split inside a `data/` folder (with `images/train`, `labels/train`, etc.) and a ready `config.yaml`, **you can skip steps 1 and 2** and go straight to Step 3. You can download the dataset from [here](https://drive.google.com/drive/folders/1XE0gM8nrx_G5SvbS96ajqq9MPITGu2-7?usp=sharing).

1. **Prepare Data** *(Optional)*: If you only have raw images in an `images/` folder and a `data.csv` file, run the reformatter to convert and split the data into YOLO format.
   ```bash
   python reformatter.py
   ```
2. **Generate Configuration** *(Optional)*: Create the `config.yaml` with dataset details.
   ```bash
   python create_config_file.py
   ```
3. **Train Models**: Execute the training scripts via Modal to run them on cloud GPUs.
   ```bash
   modal run yolov8l.py
   # OR
   modal run yolov10m.py
   ```
4. **Visualize Predictions**: After saving the best weights locally to `runs/`, you can test and visualize inference on random test images:
   ```bash
   python visualization.py 8   # For visualizing YOLOv8 results
   # OR
   python visualization.py 10  # For visualizing YOLOv10 results
   ```
5. **View Metrics**: Update the path in `metrics.py` to point to a specific model's `results.csv` to export its metrics summary.
   ```bash
   python metrics.py
   ```
