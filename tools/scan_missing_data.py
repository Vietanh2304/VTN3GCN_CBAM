"""
scan_missing_data.py

Scan train/val/test CSVs and report missing video files by checking against 
the provided video directories. Identifies FULL_3, PARTIAL_2, PARTIAL_1, 
and FULL_MISSING states for multi-view multi-stream ISLR datasets.
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm

def scan_split(split_name, csv_path, video_dir, report_file):
    if not os.path.exists(csv_path):
        msg = f"=== SPLIT: {split_name} ===\nCSV not found: {csv_path}\n\n"
        print(msg)
        report_file.write(msg)
        return

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    full_3 = 0
    partial_2 = 0
    partial_1 = 0
    full_missing = 0
    
    missing_all_ids = []
    partial_missing_info = []
    
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc=f"Scanning {split_name}"):
        views = {'center': row['center'], 'left': row['left'], 'right': row['right']}
        found_count = 0
        missing_views = []
        
        for view_name, file_name in views.items():
            if pd.isna(file_name):
                missing_views.append(view_name)
                continue
                
            file_path = os.path.join(video_dir, file_name)
            if os.path.exists(file_path):
                found_count += 1
            else:
                missing_views.append(view_name)
                
        if found_count == 3:
            full_3 += 1
        elif found_count == 2:
            partial_2 += 1
            if len(partial_missing_info) < 20:
                partial_missing_info.append(f"{row['ID']} (missing: {', '.join(missing_views)})")
        elif found_count == 1:
            partial_1 += 1
            if len(partial_missing_info) < 20:
                partial_missing_info.append(f"{row['ID']} (missing: {', '.join(missing_views)})")
        else:
            full_missing += 1
            if len(missing_all_ids) < 20:
                missing_all_ids.append(str(row['ID']))

    # Format report
    report_lines = [
        f"=== SPLIT: {split_name} ===",
        f"Total rows: {total_rows}",
        f"FULL_3:        {full_3:<5} ({full_3/total_rows*100:.2f}%)",
        f"PARTIAL_2:     {partial_2:<5} ({partial_2/total_rows*100:.2f}%)",
        f"PARTIAL_1:     {partial_1:<5} ({partial_1/total_rows*100:.2f}%)",
        f"FULL_MISSING:  {full_missing:<5} ({full_missing/total_rows*100:.2f}%)",
        f"First 20 fully-missing IDs: [{', '.join(missing_all_ids)}]",
        f"First 20 partial-missing samples (ID + which views missing): ["
    ]
    
    for info in partial_missing_info:
        report_lines.append(f"  \"{info}\",")
    report_lines.append("]\n")
    
    report_text = "\n".join(report_lines)
    print(report_text)
    report_file.write(report_text + "\n")

def main():
    parser = argparse.ArgumentParser(description="Scan CSVs and report missing video files")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing train/val/test_labels.csv")
    parser.add_argument("--train_video_dir", type=str, required=True, help="Directory for train videos")
    parser.add_argument("--val_video_dir", type=str, required=True, help="Directory for val videos")
    parser.add_argument("--test_video_dir", type=str, required=True, help="Directory for test videos")
    parser.add_argument("--output_report", type=str, default="missing_report.txt", help="Output report file")
    
    args = parser.parse_args()
    
    splits = [
        ("train", "train_labels.csv", args.train_video_dir),
        ("val", "val_labels.csv", args.val_video_dir),
        ("test", "test_labels.csv", args.test_video_dir)
    ]
    
    with open(args.output_report, 'w', encoding='utf-8') as f:
        for split_name, csv_name, video_dir in splits:
            csv_path = os.path.join(args.label_dir, csv_name)
            scan_split(split_name, csv_path, video_dir, f)

if __name__ == "__main__":
    main()
