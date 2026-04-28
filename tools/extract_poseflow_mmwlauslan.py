"""
Original File: tools/extract_poseflow.py
Adapted For: MM-WLAuslan (3 views: center/left/right)
Logic: EXACTLY the same 5 core functions. 
Calculates optical flow-like features from keypoints.
"""

import argparse
import glob
import json
import math
import os
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def read_pose(kp_file):
    with open(kp_file) as kf:
        value = json.loads(kf.read())
        kps = np.array(value['wholebody_threshold_02'])
        x = kps[:,0]
        y = kps[:,1]
        return np.stack((x, y), axis=1)

def read_neck_and_head_top(kp_file):
    with open(kp_file) as kf:
        value = json.loads(kf.read())
        kps = np.array(value['pose_threshold_02'])
        x = kps[:,0]
        y = kps[:,1]
        return np.stack((x, y), axis=1)[17:19]

def calc_pose_flow(prev, next):
    result = np.zeros_like(prev)
    for kpi in range(prev.shape[0]):
        if np.count_nonzero(prev[kpi]) == 0 or np.count_nonzero(next[kpi]) == 0:
            result[kpi, 0] = 0.0
            result[kpi, 1] = 0.0
            continue

        ang = math.atan2(next[kpi, 1] - prev[kpi, 1], next[kpi, 0] - prev[kpi, 0])
        mag = np.linalg.norm(next[kpi] - prev[kpi])

        result[kpi, 0] = ang
        result[kpi, 1] = mag

    return result

def impute_missing_keypoints(poses):
    """Replace missing keypoints (on the origin) by values from neighbouring frames."""
    # 1. Collect missing keypoints
    missing_keypoints = defaultdict(list)  # frame index -> keypoint indices that are missing
    for i in range(poses.shape[0]):
        for kpi in range(poses.shape[1]):
            if np.count_nonzero(poses[i, kpi]) == 0:  # Missing keypoint at (0, 0)
                missing_keypoints[i].append(kpi)
    # 2. Impute them
    for i in missing_keypoints.keys():
        missing = missing_keypoints[i]
        for kpi in missing:
            # Possible replacements
            candidates = poses[:, kpi]
            min_dist = np.inf
            replacement = -1
            for f in range(candidates.shape[0]):
                if f != i and np.count_nonzero(candidates[f]) > 0:
                    distance = abs(f - i)
                    if distance < min_dist:
                        min_dist = distance
                        replacement = f
            # Replace
            if replacement > -1:
                poses[i, kpi] = poses[replacement, kpi]
    # 3. We have imputed as many keypoints as possible with the closest non-missing temporal neighbours
    return poses


def normalize(poses,neck_and_head_top):
    """Normalize each pose in the array to account for camera position. We normalize
    by dividing keypoints by a factor such that the length of the neck becomes 1."""
    new_poses = []
    for i in range(poses.shape[0]):
        upper_neck = neck_and_head_top[i][1]
        head_top = neck_and_head_top[i][0]
        neck_length = np.linalg.norm(upper_neck - head_top)
        poses[i] /= neck_length
        upper_neck  /= neck_length
        head_top /= neck_length
        new_pose = np.zeros((135,2))
        new_pose[:17] = poses[i][:17]
        new_pose[17] = upper_neck
        new_pose[18] = head_top
        new_pose[19:] = poses[i][17:]
        new_poses.append(new_pose)
        assert math.isclose(np.linalg.norm(upper_neck - head_top), 1)
    return np.stack(new_poses,axis = 0)

def main():
    parser = argparse.ArgumentParser(description='Extract poseflow for MM-WLAuslan')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV with center, left, right columns')
    parser.add_argument('--poses_dir', type=str, required=True, help='Directory containing poses JSONs')
    parser.add_argument('--wholebody_dir', type=str, required=True, help='Directory containing wholebody JSONs')
    parser.add_argument('--output_dir', type=str, default='/mnt/sda1/VSLR_Storage/MM-WLAuslan/poseflow', help='Output directory for NPYs')
    parser.add_argument('--shard', type=int, default=0, help='Shard index')
    parser.add_argument('--num_shards', type=int, default=1, help='Total shards')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = df.iloc[args.shard::args.num_shards]
    
    views = ['center', 'left', 'right']
    
    # Collect videos to process
    videos_to_process = []
    for idx, row in df.iterrows():
        for view in views:
            if not pd.isna(row[view]):
                videos_to_process.append(row[view].replace('.mp4', ''))
                
    with tqdm(total=len(videos_to_process), desc=f"Shard {args.shard}/{args.num_shards}") as pbar:
        for file_name in videos_to_process:
            input_dir_poses = os.path.join(args.poses_dir, file_name)
            input_dir_wholebody = os.path.join(args.wholebody_dir, file_name)
            output_dir = os.path.join(args.output_dir, file_name)
            
            if not os.path.exists(input_dir_poses) or not os.path.exists(input_dir_wholebody):
                print(f"WARNING: Missing poses or wholebody for {file_name}")
                pbar.update(1)
                continue
                
            kp_files_poses = sorted(glob.glob(os.path.join(input_dir_poses, '*.json')))
            kp_files_wholebody = sorted(glob.glob(os.path.join(input_dir_wholebody, '*.json')))
            
            # Use wholebody count since read_pose reads from wholebody
            n_json = len(kp_files_wholebody)
            
            if os.path.exists(output_dir):
                flow_files = glob.glob(os.path.join(output_dir, 'flow_*.npy'))
                if len(flow_files) >= max(0, n_json - 1):
                    pbar.update(1)
                    continue
                else:
                    # Partial computation detected → clean up for fresh recompute
                    shutil.rmtree(output_dir)
                    
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                poses = []
                neck_and_head_top = []
                # Ensure we match the pairs correctly. Assume they have same length and index matching
                # as generated by frame count. We should use zip or ensure lengths match.
                min_len = min(len(kp_files_poses), len(kp_files_wholebody))
                for i in range(min_len):
                    poses.append(read_pose(kp_files_wholebody[i]))
                    neck_and_head_top.append(read_neck_and_head_top(kp_files_poses[i]))
                    
                if len(poses) == 0:
                    print(f"WARNING: Empty poses for {file_name}")
                    pbar.update(1)
                    continue
                    
                poses = np.stack(poses)
                neck_and_head_top = np.stack(neck_and_head_top)
                poses = impute_missing_keypoints(poses)
                neck_and_head_top = impute_missing_keypoints(neck_and_head_top)
                poses = normalize(poses, neck_and_head_top)

                # 2. Compute pose flow
                prev = poses[0]
                for i in range(1, poses.shape[0]):
                    next = poses[i]
                    flow = calc_pose_flow(prev, next)
                    np.save(os.path.join(output_dir, 'flow_{:05d}'.format(i - 1)), flow)
                    prev = next
            except Exception as e:
                print(f"WARNING: Exception processing flow for {file_name}: {str(e)}")
                
            pbar.update(1)

if __name__ == '__main__':
    main()
