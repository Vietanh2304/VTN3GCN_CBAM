import cv2
import numpy as np
import os
import torch
import glob
import re

def gen_gaussian_hmap_op(coords, raw_size=(260, 210), map_size=None, sigma=1, confidence=False, threshold=0, **kwargs):
    # Existing implementation of the heatmap generation function
    T, hmap_num = coords.shape[:2]
    raw_h, raw_w = raw_size
    if map_size == None:
        map_h, map_w = raw_h, raw_w
        factor_h, factor_w = 1, 1
    else:
        map_h, map_w = map_size
        factor_h, factor_w = map_h / raw_h, map_w / raw_w
    # generate 2d coords
    coords_y = coords[..., 1] * factor_h
    coords_x = coords[..., 0] * factor_w
    confs = coords[..., 2]  # T, C

    y, x = torch.meshgrid(torch.arange(map_h), torch.arange(map_w), indexing='ij')
    coords = torch.stack([coords_y, coords_x], dim=0)
    grid = torch.stack([y, x], dim=0).to(coords.device)  # [2,H,W]
    grid = grid.unsqueeze(0).unsqueeze(0).expand(hmap_num, T, -1, -1, -1)  # [C,T,2,H,W]
    coords = coords.unsqueeze(0).unsqueeze(0).expand(map_h, map_w, -1, -1, -1).permute(4, 3, 2, 0, 1)  # [C,T,2,H,W]
    hmap = torch.exp(-((grid - coords) ** 2).sum(dim=2) / (2 * sigma ** 2))  # [C,T,H,W]
    hmap = hmap.permute(1, 0, 2, 3)  # [T,C,H,W]
    if confidence:
        confs = confs.unsqueeze(-1).unsqueeze(-1)  # T,C,1,1
        confs = torch.where(confs > threshold, confs, torch.zeros_like(confs))
        hmap = hmap * confs

    return hmap


def get_frame_number(filename):
    # Extract the frame number from the filename
    match = re.search(r'hand_kp_(\d+)\.npy', filename)
    if match:
        return int(match.group(1))
    else:
        return -1  # Assign -1 if no frame number is found


def process_video_with_keypoints(video_path, keypoint_files, save_dir):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(keypoint_files) != total_frames:
        print(
            f"Warning: Number of keypoint files ({len(keypoint_files)}) does not match number of video frames ({total_frames}) for video '{os.path.basename(video_path)}'.")

    frame_idx = 0
    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))

        if frame_idx < len(keypoint_files):
            keypoint_file = keypoint_files[frame_idx]
            data = np.load(keypoint_file)  # Data is a NumPy array of shape (46, 2)

            # Check that data shape is correct
            if data.shape != (46, 2):
                print(f"Unexpected data shape in {keypoint_file}: {data.shape}. Skipping this frame.")
                frame_idx += 1
                continue

            keypoints = data  # Shape: (46, 2)
            confidences = np.ones(46)  # Set default confidence to 1.0 for all keypoints
        else:
            print(f"No keypoint data for frame {frame_idx}. Skipping.")
            frame_idx += 1
            continue

        # Convert normalized coordinates to image coordinates
        img_h, img_w = frame.shape[:2]
        x_coords = keypoints[:, 0] * img_w
        y_coords = keypoints[:, 1] * img_h

        # Prepare coords tensor for gen_gaussian_hmap_op
        coords = np.stack([x_coords, y_coords, confidences], axis=-1)[np.newaxis, :, :]  # Shape: (1, 46, 3)
        coords_tensor = torch.from_numpy(coords).float()

        # Generate heatmap
        hmap = gen_gaussian_hmap_op(
            coords_tensor, raw_size=(img_h, img_w), map_size=None,
            sigma=4, confidence=True, threshold=0.5)

        # Sum heatmaps over keypoints
        heatmap = hmap.sum(dim=1)[0]  # Shape: [H, W]

        # Normalize heatmap to 0-255
        heatmap_np = heatmap.cpu().numpy()
        if np.max(heatmap_np) > 0:
            heatmap_np = (heatmap_np / np.max(heatmap_np) * 255).astype(np.uint8)
        else:
            heatmap_np = (heatmap_np * 255).astype(np.uint8)

        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

        # Overlay heatmap on frame
        overlay = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)

        # Save the frame with heatmap overlay
        output_file = os.path.join(save_dir, f"heatmap_{frame_idx:05d}.jpg")
        cv2.imwrite(output_file, overlay)

        frame_idx += 1

    cap.release()


def process_videos_with_keypoints(videos_folder, keypoints_folder, output_folder):
    # Get all video files in the videos folder
    video_files = glob.glob(os.path.join(videos_folder, '*.mp4'))

    if not video_files:
        print(f"No video files found in {videos_folder}")
        return

    for video_file in video_files:
        # Get the base name of the video without extension
        video_name = os.path.splitext(os.path.basename(video_file))[0]

        # Construct the path to the keypoints subfolder
        keypoints_subfolder = os.path.join(keypoints_folder, video_name)

        # Check if the keypoints subfolder exists
        if not os.path.isdir(keypoints_subfolder):
            print(f"Keypoints folder '{keypoints_subfolder}' does not exist for video '{video_name}'. Skipping.")
            continue

        # Find all .npy files in the keypoints subfolder
        keypoint_pattern = os.path.join(keypoints_subfolder, 'hand_kp_*.npy')
        keypoint_files = glob.glob(keypoint_pattern)

        # Sort the keypoint files numerically based on frame number
        keypoint_files.sort(key=lambda f: get_frame_number(f))

        if not keypoint_files:
            print(f"No keypoint files found in '{keypoints_subfolder}'. Skipping.")
            continue

        # Create the output directory for this video
        save_directory = os.path.join(output_folder, video_name)
        if os.path.exists(save_directory):
            print(f"Output directory for video '{video_name}' already exists. Skipping processing.")
            continue

        os.makedirs(save_directory, exist_ok=True)

        print(f"Processing video: {video_name}")
        process_video_with_keypoints(video_file, keypoint_files, save_directory)
        print(f"Finished processing {video_name}")

if __name__ == "__main__":
    videos_folder = "/media/ibmelab/ibme21/GG/VSLRecognition/vsl/videos"  # Replace with the path to your videos folder
    keypoints_folder = "/media/ibmelab/ibme21/GG/VSLRecognition/vsl/hand_keypoints"  # Replace with the path to your .npy keypoints folder
    output_folder = "/media/ibmelab/ibme21/GG/VSLRecognition/vsl/heatmap"  # Replace with the path to your output folder
    process_videos_with_keypoints(videos_folder, keypoints_folder, output_folder)
    print("All videos have been processed.")
