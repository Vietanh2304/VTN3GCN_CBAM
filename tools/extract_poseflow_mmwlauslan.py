"""
extract_poseflow_mmwlauslan.py

Compute optical-flow-like poseflow features from COCO-WholeBody keypoints.
Reads from wholebody HDF5 (Phase 4 output), writes to poseflow HDF5.

IMPORTANT: The 5 core algorithmic functions (read_pose, calc_pose_flow,
impute_missing_keypoints, normalize, read_neck_and_head_top) are UNCHANGED
from the original extract_poseflow.py.
"""

import argparse
import math
import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────────────
# CORE ALGORITHM (do NOT modify — kept identical to extract_poseflow.py)
# ─────────────────────────────────────────────────────────────────

def read_pose(frame_data):
    """Read (133, 2) xy from a (133, 3) frame array."""
    kps = frame_data  # (133, 3)
    x = kps[:, 0]
    y = kps[:, 1]
    return np.stack((x, y), axis=1)  # (133, 2)


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
    missing_keypoints = defaultdict(list)
    for i in range(poses.shape[0]):
        for kpi in range(poses.shape[1]):
            if np.count_nonzero(poses[i, kpi]) == 0:
                missing_keypoints[i].append(kpi)
    for i in missing_keypoints.keys():
        missing = missing_keypoints[i]
        for kpi in missing:
            candidates = poses[:, kpi]
            min_dist = np.inf
            replacement = -1
            for f in range(candidates.shape[0]):
                if f != i and np.count_nonzero(candidates[f]) > 0:
                    distance = abs(f - i)
                    if distance < min_dist:
                        min_dist = distance
                        replacement = f
            if replacement > -1:
                poses[i, kpi] = poses[replacement, kpi]
    return poses


def normalize_by_neck_dist(poses):
    """Normalize each pose by neck-length so flow is scale-invariant.

    For COCO-WholeBody 133-kp:
      - neck is approximated as midpoint of left_shoulder (5) and right_shoulder (6)
      - head top is approximated as nose (0)
    Returns the normalized poses array (same shape, no extra appended rows).
    """
    new_poses = []
    for i in range(poses.shape[0]):
        neck = (poses[i][5] + poses[i][6]) / 2.0
        head_top = poses[i][0]
        neck_length = np.linalg.norm(neck - head_top)
        if neck_length < 1e-6:
            new_poses.append(poses[i].copy())
            continue
        norm_pose = poses[i] / neck_length
        new_poses.append(norm_pose)
    return np.stack(new_poses, axis=0)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Extract poseflow for MM-WLAuslan (HDF5 I/O)')
    parser.add_argument('--input_h5', type=str, required=True,
                        help='Input wholebody HDF5 (e.g. /mnt/.../wholebody_h5/train.h5)')
    parser.add_argument('--output_h5', type=str, required=True,
                        help='Output poseflow HDF5 (e.g. /mnt/.../poseflow_h5/train.h5)')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.input_h5):
        print(f"ERROR: Input HDF5 not found: {args.input_h5}")
        return

    os.makedirs(os.path.dirname(args.output_h5), exist_ok=True)

    total_processed = 0
    total_skipped = 0
    total_failed = 0

    with h5py.File(args.input_h5, 'r') as f_in, \
         h5py.File(args.output_h5, 'a') as f_out:

        all_video_ids = list(f_in.keys())
        # Shard
        shard_ids = all_video_ids[args.shard::args.num_shards]

        for video_id in tqdm(shard_ids, desc=f"Poseflow shard {args.shard}/{args.num_shards}"):
            # Resume: skip if already done
            if video_id in f_out and 'poseflow' in f_out[video_id]:
                total_skipped += 1
                continue

            group_in = f_in[video_id]
            if 'wholebody_threshold_02' not in group_in:
                print(f"WARNING: No wholebody_threshold_02 for {video_id}, skipping.")
                total_failed += 1
                continue

            try:
                # Load all frames: (T, 133, 3)
                wb_data = group_in['wholebody_threshold_02'][:]
                T = wb_data.shape[0]

                if T < 2:
                    print(f"WARNING: Too few frames ({T}) for {video_id}, skipping.")
                    total_failed += 1
                    continue

                # Build (T, 133, 2) xy poses
                poses = np.stack([read_pose(wb_data[t]) for t in range(T)], axis=0)

                # Impute and normalize
                poses = impute_missing_keypoints(poses)
                poses = normalize_by_neck_dist(poses)

                # Compute poseflow: (T-1, 133, 2)
                flow_list = []
                prev = poses[0]
                for i in range(1, T):
                    nxt = poses[i]
                    flow = calc_pose_flow(prev, nxt)   # (133, 2)
                    flow_list.append(flow)
                    prev = nxt

                poseflow_arr = np.array(flow_list, dtype=np.float32)  # (T-1, 133, 2)

                # Write to HDF5
                if video_id in f_out:
                    del f_out[video_id]   # clean partial group
                grp = f_out.create_group(video_id)
                grp.create_dataset(
                    'poseflow',
                    data=poseflow_arr,
                    compression='gzip',
                    compression_opts=4,
                    chunks=(1, 133, 2)
                )
                total_processed += 1

            except Exception as e:
                print(f"WARNING: Exception for {video_id}: {e}")
                total_failed += 1

    print(f"\n--- Poseflow Shard {args.shard} Summary ---")
    print(f"Total processed : {total_processed}")
    print(f"Total skipped   : {total_skipped}")
    print(f"Total failed    : {total_failed}")


if __name__ == '__main__':
    main()
