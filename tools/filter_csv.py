"""
filter_csv.py

Filter train/val/test CSVs to KEEP only rows where all 3 view video files 
exist on the disk. Outputs clean CSVs to the specified output directory.
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm

def filter_split(split_name, csv_path, video_dir, output_dir):
    if not os.path.exists(csv_path):
        print(f"[{split_name}] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    keep_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc=f"Filtering {split_name}"):
        views = [row['center'], row['left'], row['right']]
        found_all = True
        
        for file_name in views:
            if pd.isna(file_name) or not os.path.exists(os.path.join(video_dir, file_name)):
                found_all = False
                break
                
        if found_all:
            keep_indices.append(idx)
            
    df_clean = df.loc[keep_indices].copy()
    
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{split_name}_labels_clean.csv")
    
    # Ensure column order is preserved (center, left, right, ID, label_id)
    cols = ['center', 'left', 'right', 'ID', 'label_id']
    df_clean = df_clean[cols]
    
    df_clean.to_csv(out_csv, index=False)
    
    kept = len(df_clean)
    removed = total_rows - kept
    print(f"[{split_name.capitalize()}] {kept} kept / {total_rows} total ({removed} removed). Saved to {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Filter CSVs to keep only FULL_3 rows")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing train/val/test_labels.csv")
    parser.add_argument("--train_video_dir", type=str, required=True, help="Directory for train videos")
    parser.add_argument("--val_video_dir", type=str, required=True, help="Directory for val videos")
    parser.add_argument("--test_video_dir", type=str, required=True, help="Directory for test videos")
    parser.add_argument("--output_dir", type=str, default="/mnt/sda1/VSLR_Storage/MM-WLAuslan/labels_clean", help="Directory for output clean CSVs")
    
    args = parser.parse_args()
    
    splits = [
        ("train", "train_labels.csv", args.train_video_dir),
        ("val", "val_labels.csv", args.val_video_dir),
        ("test", "test_labels.csv", args.test_video_dir)
    ]
    
    for split_name, csv_name, video_dir in splits:
        csv_path = os.path.join(args.label_dir, csv_name)
        filter_split(split_name, csv_path, video_dir, args.output_dir)

if __name__ == "__main__":
    main()
