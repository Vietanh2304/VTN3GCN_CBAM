"""
Original File: tools/gen_wholebody.py
Adapted For: MM-WLAuslan (3 views: center/left/right)
Model: td-hm_res152_8xb32-210e_coco-wholebody-384x288
Outputs: 133 keypoints (COCO-WholeBody format), keys: raw_wholebody, wholebody_threshold_02, prob
"""

import argparse
import json
import os
import cv2
import pandas as pd
from tqdm.auto import tqdm
import warnings

def gen_wholebody(video_url, kp_folder, wholebody_detector, file_name):
    if not os.path.exists(video_url):
        print(f"WARNING: Video not found {video_url}")
        return

    # Check resume logic
    cap = cv2.VideoCapture(video_url)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if os.path.exists(kp_folder):
        num_existing = len([f for f in os.listdir(kp_folder) if f.endswith('.json')])
        if num_existing >= frame_count and frame_count > 0:
            return  # Skip if already done
            
    os.makedirs(kp_folder, exist_ok=True)
    
    try:
        wholebody_results = wholebody_detector(video_url)
        for idx, wholebody_result in enumerate(wholebody_results):
            if 'predictions' not in wholebody_result or len(wholebody_result['predictions']) == 0 or len(wholebody_result['predictions'][0]) == 0:
                continue
                
            wholebody = wholebody_result['predictions'][0][0]['keypoints']
            prob = wholebody_result['predictions'][0][0]['keypoint_scores']
            
            raw_wholebody = [[value[0], value[1], 0] for idx, value in enumerate(wholebody)]
            wholebody_threshold_02 = [[value[0], value[1], 0] if prob[idx] > 0.2 else [0, 0, 0] for idx, value in enumerate(wholebody)]
            
            dict_data = {
                "raw_wholebody": raw_wholebody,
                "wholebody_threshold_02": wholebody_threshold_02,
                "prob": prob.tolist() if hasattr(prob, 'tolist') else list(prob)
            }
            
            dest = os.path.join(kp_folder, file_name.replace(".mp4", "") + '_{:06d}_keypoints.json'.format(idx))
            with open(dest, 'w') as f:
                json.dump(dict_data, f)
    except Exception as e:
        print(f"WARNING: Exception processing {video_url}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Extract wholebody for MM-WLAuslan')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV with center, left, right columns')
    parser.add_argument('--video_dir', type=str, required=True, help='Base directory for videos')
    parser.add_argument('--output_dir', type=str, default='/mnt/sda1/VSLR_Storage/MM-WLAuslan/wholebody', help='Output directory for JSONs')
    parser.add_argument('--shard', type=int, default=0, help='Shard index')
    parser.add_argument('--num_shards', type=int, default=1, help='Total shards')
    args = parser.parse_args()

    # Import inside to avoid slow startup if just running --help
    from mmpose.apis import MMPoseInferencer
    wholebody_detector = MMPoseInferencer("td-hm_res152_8xb32-210e_coco-wholebody-384x288")

    df = pd.read_csv(args.csv_path)
    df = df.iloc[args.shard::args.num_shards]
    
    views = ['center', 'left', 'right']
    total_tasks = len(df) * len(views)
    
    with tqdm(total=total_tasks, desc=f"Shard {args.shard}/{args.num_shards}") as pbar:
        for idx, row in df.iterrows():
            for view in views:
                file_name = row[view]
                if pd.isna(file_name):
                    pbar.update(1)
                    continue
                    
                video_url = os.path.join(args.video_dir, file_name)
                kp_folder = os.path.join(args.output_dir, file_name.replace(".mp4", ""))
                
                gen_wholebody(video_url, kp_folder, wholebody_detector, file_name)
                pbar.update(1)

if __name__ == "__main__":
    main()
