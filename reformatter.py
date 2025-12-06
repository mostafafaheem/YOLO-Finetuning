import os
import shutil

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

columns = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]


df = pd.read_csv("data.csv", names=columns)

df_grouped = df.groupby("filename")

out_folder = "data/"
classes = {cell: i for i, cell in enumerate(df["class"].unique())}

for i in ["images", "labels"]:
    for j in ["train", "test"]:
        os.makedirs(os.path.join(out_folder, i, j), exist_ok=True)

train, test = train_test_split(df["filename"].unique(), test_size=0.2, random_state=42)

for file_list, split_name in zip([train, test], ["train", "test"]):
    for img_name in file_list:
        source_path = os.path.join("images/", img_name)
        destination_path = os.path.join(out_folder, "images", split_name, img_name)

        shutil.copy(source_path, destination_path)

        img = cv2.imread(source_path)
        if img is None:
            continue
        height = img.shape[0]
        width = img.shape[1]

        boxes_df = df_grouped.get_group(img_name)
        label_path = os.path.join(
            out_folder, "labels", split_name, img_name.replace(".jpg", ".txt")
        )
        valid_boxes_df = boxes_df[boxes_df["class"].isin(classes.keys())].copy()

        valid_boxes_df["cls_id"] = valid_boxes_df["class"].map(classes)

        valid_boxes_df["box_w"] = valid_boxes_df["xmax"] - valid_boxes_df["xmin"]
        valid_boxes_df["box_h"] = valid_boxes_df["ymax"] - valid_boxes_df["ymin"]

        valid_boxes_df["box_cx"] = valid_boxes_df["xmin"] + (
            valid_boxes_df["box_w"] / 2
        )
        valid_boxes_df["box_cy"] = valid_boxes_df["ymin"] + (
            valid_boxes_df["box_h"] / 2
        )

        valid_boxes_df["norm_cx"] = valid_boxes_df["box_cx"] / width
        valid_boxes_df["norm_cy"] = valid_boxes_df["box_cy"] / height
        valid_boxes_df["norm_w"] = valid_boxes_df["box_w"] / width
        valid_boxes_df["norm_h"] = valid_boxes_df["box_h"] / height

        output_lines = valid_boxes_df.apply(
            lambda row: f"{int(row['cls_id'])} {row['norm_cx']:.6f} {row['norm_cy']:.6f} {row['norm_w']:.6f} {row['norm_h']:.6f}",
            axis=1,
        ).tolist()

        with open(label_path, "w") as f:
            f.write("\n".join(output_lines) + "\n")
