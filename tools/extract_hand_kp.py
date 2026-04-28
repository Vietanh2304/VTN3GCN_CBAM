import os
import json
import numpy as np
import glob
from tqdm import tqdm
import re
def extract_hand_keypoints(base_dir):
    poses_dir = os.path.join(base_dir, 'poses')
    hand_kp_dir = os.path.join(base_dir, 'hand_keypoints')
    
    if not os.path.exists(poses_dir):
        print(f"Lỗi: Không tìm thấy thư mục pose tại {poses_dir}")
        return

    os.makedirs(hand_kp_dir, exist_ok=True)
    video_folders = [f for f in os.listdir(poses_dir) if os.path.isdir(os.path.join(poses_dir, f))]
    
    for vid_name in tqdm(video_folders, desc="Extracting Hand KPs for MM-WLAuslan"):
        vid_pose_path = os.path.join(poses_dir, vid_name)
        save_vid_dir = os.path.join(hand_kp_dir, vid_name)
        os.makedirs(save_vid_dir, exist_ok=True)
        
        json_files = sorted(glob.glob(os.path.join(vid_pose_path, '*keypoints.json')))
        
        for json_file in json_files:
            try:
                match = re.search(r'(\d{5,6})', os.path.basename(json_file))
                frame_idx = int(match.group()) if match else 0
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Trích xuất 46 điểm tay (pose_threshold_02)
                kp_data = data.get('pose_threshold_02', [])
                if not kp_data or len(kp_data) == 0:
                    hand_kp_2d = np.zeros((46, 2))
                else:
                    hand_kp_2d = np.array(kp_data).reshape(-1, 3)[:, :2]
                    if hand_kp_2d.shape[0] < 46:
                        pad_size = 46 - hand_kp_2d.shape[0]
                        padding = np.zeros((pad_size, 2))
                        hand_kp_2d = np.vstack((hand_kp_2d, padding))
                    elif hand_kp_2d.shape[0] > 46:
                        hand_kp_2d = hand_kp_2d[:46, :]
                
                save_path = os.path.join(save_vid_dir, f'hand_kp_{frame_idx:05d}.npy')
                np.save(save_path, hand_kp_2d) 
            except Exception:
                continue

if __name__ == '__main__':
    base_url = "/mnt/sda1/VSLR_Storage/MM-WLAuslan"
    extract_hand_keypoints(base_url)