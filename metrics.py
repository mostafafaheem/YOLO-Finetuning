import pandas as pd


def create_metrics_report(csv_path, output_docx="Model_Report.docx"):

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]  # Clean column names

    best_row_idx = df["metrics/mAP50-95(B)"].idxmax()
    best_metrics = df.iloc[best_row_idx]

    metrics_to_show = {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP @ 50%",
        "metrics/mAP50-95(B)": "mAP @ 50-95%",
        "train/box_loss": "Training Box Loss",
        "val/box_loss": "Validation Box Loss",
    }

    # Add rows for each metric
    for col_name, display_name in metrics_to_show.items():
        if col_name in best_metrics:
            print(f"{col_name}, {best_metrics[col_name]:.4f}")
            # row_cells = table.add_row().cells
            # row_cells[0].text = display_name
            # # Format to 4 decimal places
            # row_cells[1].text =

    # --- Add a Footer/Note ---
    # doc.add_paragraph("\n")
    # note = doc.add_paragraph("Note: 'Best' weights are saved automatically by YOLO based on a weighted fitness score of these metrics.")
    # note.runs[0].font.italic = True
    # note.runs[0].font.size = Pt(9)

    # # 4. Save
    # doc.save(output_docx)
    # print(f"✅ Report saved successfully to {output_docx}")


create_metrics_report(
    r"C:\Users\mosta\OneDrive\Desktop\Semester 7\Computer Vision\Assignments\Assignment 2\2022170623\runs\yolov8l_run\results.csv"
)
