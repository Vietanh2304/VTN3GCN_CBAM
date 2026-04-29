"""
extract_hand_kp.py

Extract 46 hand and body keypoints from COCO-WholeBody HDF5 format for the 
AAGCN GCN branch. Uses a specific permutation to map from COCO-133 indices 
to the 46-kp format expected by the model's graph topology.
"""

import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm

# Permutation: AAGCN-46 idx -> COCO-WholeBody-133 idx
# Built by mapping AAGCN/graph.py BodyIdentifier enum names to COCO-WholeBody schema:
# COCO right hand [112:133]: 112=wrist, 113=thumb_CMC, 114=thumb_MCP, 115=thumb_IP, 116=thumb_TIP,
#                             117=index_MCP, 118=index_PIP, 119=index_DIP, 120=index_TIP,
#                             121=middle_MCP, 122=middle_PIP, 123=middle_DIP, 124=middle_TIP,
#                             125=ring_MCP, 126=ring_PIP, 127=ring_DIP, 128=ring_TIP,
#                             129=pinky_MCP, 130=pinky_PIP, 131=pinky_DIP, 132=pinky_TIP
# COCO left hand [91:112]: same layout but offset 91
# COCO body: 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist

AAGCN46_TO_COCO133 = [
    # Right hand fingers (AAGCN 0-19) — order: INDEX, MIDDLE, PINKY, RING, THUMB; per finger: DIP, MCP, PIP, TIP
    119, 117, 118, 120,   # 0: INDEX_FINGER_DIP_right, 1: MCP, 2: PIP, 3: TIP
    123, 121, 122, 124,   # 4: MIDDLE_FINGER_DIP_right, 5: MCP, 6: PIP, 7: TIP
    131, 129, 130, 132,   # 8: PINKY_DIP_right, 9: MCP, 10: PIP, 11: TIP
    127, 125, 126, 128,   # 12: RING_FINGER_DIP_right, 13: MCP, 14: PIP, 15: TIP
    113, 115, 114, 116,   # 16: THUMB_CMC_right, 17: IP, 18: MCP, 19: TIP
    112,                  # 20: WRIST_right
    # Left hand fingers (AAGCN 21-40) — same order, COCO left hand offset 91
    98, 96, 97, 99,       # 21: INDEX_FINGER_DIP_left, 22: MCP, 23: PIP, 24: TIP
    102, 100, 101, 103,   # 25: MIDDLE_FINGER_DIP_left, ...
    110, 108, 109, 111,   # 29: PINKY_DIP_left, ...
    106, 104, 105, 107,   # 33: RING_FINGER_DIP_left, ...
    92, 94, 93, 95,       # 37: THUMB_CMC_left, IP, MCP, TIP
    91,                   # 41: WRIST_left
    # Body (AAGCN 42-45)
    6,  # 42: RIGHT_SHOULDER → COCO 6
    5,  # 43: LEFT_SHOULDER → COCO 5
    7,  # 44: LEFT_ELBOW → COCO 7
    8,  # 45: RIGHT_ELBOW → COCO 8
]
assert len(AAGCN46_TO_COCO133) == 46


def extract_hand_keypoints(input_h5_path, output_h5_path):
    if not os.path.exists(input_h5_path):
        print(f"Error: Input HDF5 not found at {input_h5_path}")
        return

    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    
    total_videos = 0
    total_zeros = 0
    total_elements = 0

    with h5py.File(input_h5_path, 'r') as f_in, h5py.File(output_h5_path, 'a') as f_out:
        video_ids = list(f_in.keys())
        
        for video_id in tqdm(video_ids, desc="Extracting Hand KPs"):
            if video_id in f_out:
                # Check for completeness if resuming
                if 'hand_kp' in f_out[video_id]:
                    # Assume complete for now, or could check shapes
                    continue
                else:
                    del f_out[video_id]
            
            group_in = f_in[video_id]
            if 'wholebody_threshold_02' not in group_in:
                continue
                
            wb = group_in['wholebody_threshold_02'][:]  # (T, 133, 3)
            
            # Extract 46 keypoints using the fancy indexing permutation, take only (x, y)
            hand_kp = wb[:, AAGCN46_TO_COCO133, :2]  # (T, 46, 2)
            hand_kp = hand_kp.astype(np.float32)
            
            # Stats for non-zero ratio
            total_elements += hand_kp.size
            total_zeros += np.count_nonzero(hand_kp == 0)
            
            grp_out = f_out.create_group(video_id)
            grp_out.create_dataset("hand_kp", data=hand_kp, compression="gzip", compression_opts=4, chunks=(1, 46, 2))
            
            total_videos += 1
            
    if total_elements > 0:
        non_zero_ratio = 1.0 - (total_zeros / total_elements)
    else:
        non_zero_ratio = 0.0
        
    print("\n--- Hand Keypoint Extraction Summary ---")
    print(f"Total videos processed: {total_videos}")
    print(f"Average non-zero ratio: {non_zero_ratio:.2%}")

def main():
    parser = argparse.ArgumentParser(description="Extract hand keypoints to HDF5")
    parser.add_argument("--input_h5", type=str, required=True, help="Input wholebody HDF5 path")
    parser.add_argument("--output_h5", type=str, required=True, help="Output hand kp HDF5 path")
    
    args = parser.parse_args()
    extract_hand_keypoints(args.input_h5, args.output_h5)

if __name__ == "__main__":
    main()