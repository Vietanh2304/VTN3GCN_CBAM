import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import os
from collections import defaultdict
import torch

# Define hand and pose landmarks as per your specification
hand_landmarks = [
    'INDEX_FINGER_DIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_TIP',
    'PINKY_DIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_TIP',
    'RING_FINGER_DIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_TIP',
    'THUMB_CMC', 'THUMB_IP', 'THUMB_MCP', 'THUMB_TIP', 'WRIST'
]

HAND_IDENTIFIERS = [id + "_right" for id in hand_landmarks] + [id + "_left" for id in hand_landmarks]
POSE_IDENTIFIERS = ["RIGHT_SHOULDER", "LEFT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW"]
body_identifiers = HAND_IDENTIFIERS + POSE_IDENTIFIERS  # Total of 46 keypoints

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to find the index of the first non-zero element
def find_index(array):
    for i, num in enumerate(array):
        if num != 0:
            return i
    return -1  # Return -1 if no non-zero element is found

# Function to fill in missing keypoints
def curl_skeleton(array):
    array = list(array)
    if sum(array) == 0:
        return array
    for i, location in enumerate(array):
        if location != 0:
            continue
        else:
            if i == 0 or i == len(array) - 1:
                continue
            else:
                if array[i + 1] != 0:
                    array[i] = float((array[i - 1] + array[i + 1]) / 2)
                else:
                    j = find_index(array[i + 1:])
                    if j == -1:
                        continue
                    array[i] = float(((1 + j) * array[i - 1] + array[i + 1 + j]) / (2 + j))
    return array

def gen_gaussian_hmap_op(coords, raw_size=(260,210), map_size=None, sigma=1, confidence=False, threshold=0, **kwargs):
    # openpose version
    # pose [T,18,3]; face [T,70,3]; hand_0(left) [T,21,3]; hand_1(right) [T,21,3]
    # gamma: hyper-param, control the width of gaussian, larger gamma, SMALLER gaussian
    # flags: use pose or face or hands or some of them

    #coords T, C, 3

    T, hmap_num = coords.shape[:2]
    raw_h, raw_w = raw_size #260,210
    if map_size==None:
        map_h, map_w = raw_h, raw_w
        factor_h, factor_w = 1, 1
    else:
        map_h, map_w = map_size
        factor_h, factor_w = map_h/raw_h, map_w/raw_w
    # generate 2d coords
    # NOTE: openpose generate opencv-style coordinates!
    coords_y =  coords[..., 1]*factor_h
    coords_x = coords[..., 0]*factor_w
    confs = coords[..., 2] #T, C

    y, x = torch.meshgrid(torch.arange(map_h), torch.arange(map_w))
    coords = torch.stack([coords_y, coords_x], dim=0)
    grid = torch.stack([y,x], dim=0).to(coords.device)  #[2,H,W]
    grid = grid.unsqueeze(0).unsqueeze(0).expand(hmap_num,T,-1,-1,-1)  #[C,T,2,H,W]
    coords = coords.unsqueeze(0).unsqueeze(0).expand(map_h, map_w,-1,-1,-1).permute(4,3,2,0,1)  #[C,T,2,H,W]
    hmap = torch.exp(-((grid-coords)**2).sum(dim=2) / (2*sigma**2))  #[C,T,H,W]
    hmap = hmap.permute(1,0,2,3)  #[T,C,H,W]
    if confidence:
        confs = confs.unsqueeze(-1).unsqueeze(-1) #T,C,1,1
        confs = torch.where(confs>threshold, confs, torch.zeros_like(confs))
        hmap = hmap*confs

    return hmap

def process_video(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    mp_holistic_instance = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Prepare a dictionary to store keypoints and confidences
    keypoint_data = defaultdict(list)
    confidence_data = defaultdict(list)
    frame_count = 0

    with mp_holistic_instance as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Process right hand
            if results.right_hand_landmarks:
                # Since individual landmark confidences are not available, use hand detection confidence
                hand_confidence = results.right_hand_landmarks  # No confidence attribute; consider using a default value
                default_hand_confidence = 1.0  # You can adjust this value based on your needs
                for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                    keypoint_data[f"{hand_landmarks[idx]}_right_x"].append(landmark.x)
                    keypoint_data[f"{hand_landmarks[idx]}_right_y"].append(landmark.y)
                    confidence_data[f"{hand_landmarks[idx]}_right"].append(default_hand_confidence)
            else:
                for idx in range(len(hand_landmarks)):
                    keypoint_data[f"{hand_landmarks[idx]}_right_x"].append(0)
                    keypoint_data[f"{hand_landmarks[idx]}_right_y"].append(0)
                    confidence_data[f"{hand_landmarks[idx]}_right"].append(0)

            # Process left hand
            if results.left_hand_landmarks:
                hand_confidence = results.left_hand_landmarks  # No confidence attribute; consider using a default value
                default_hand_confidence = 1.0
                for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                    keypoint_data[f"{hand_landmarks[idx]}_left_x"].append(landmark.x)
                    keypoint_data[f"{hand_landmarks[idx]}_left_y"].append(landmark.y)
                    confidence_data[f"{hand_landmarks[idx]}_left"].append(default_hand_confidence)
            else:
                for idx in range(len(hand_landmarks)):
                    keypoint_data[f"{hand_landmarks[idx]}_left_x"].append(0)
                    keypoint_data[f"{hand_landmarks[idx]}_left_y"].append(0)
                    confidence_data[f"{hand_landmarks[idx]}_left"].append(0)

            # Process pose landmarks (shoulders and elbows)
            if results.pose_landmarks:
                for pose_identifier in POSE_IDENTIFIERS:
                    idx_pose = getattr(mp_holistic.PoseLandmark, pose_identifier).value
                    landmark = results.pose_landmarks.landmark[idx_pose]
                    keypoint_data[f"{pose_identifier}_x"].append(landmark.x)
                    keypoint_data[f"{pose_identifier}_y"].append(landmark.y)
                    confidence_data[f"{pose_identifier}"].append(landmark.visibility)
            else:
                for pose_identifier in POSE_IDENTIFIERS:
                    keypoint_data[f"{pose_identifier}_x"].append(0)
                    keypoint_data[f"{pose_identifier}_y"].append(0)
                    confidence_data[f"{pose_identifier}"].append(0)

    # Process the keypoints and confidences
    T = frame_count  # Number of frames processed
    num_keypoints = len(body_identifiers)
    keypoints_all_frames = np.empty((T, num_keypoints, 2))
    confidences_all_frames = np.empty((T, num_keypoints))

    for index, identifier in enumerate(body_identifiers):
        x_key = identifier + "_x"
        y_key = identifier + "_y"
        conf_key = identifier  # Confidence keys do not have '_x' or '_y'

        x_array = keypoint_data.get(x_key, [0] * T)
        y_array = keypoint_data.get(y_key, [0] * T)
        conf_array = confidence_data.get(conf_key, [0] * T)

        data_keypoint_preprocess_x = curl_skeleton(x_array)
        data_keypoint_preprocess_y = curl_skeleton(y_array)
        data_confidence_preprocess = curl_skeleton(conf_array)

        keypoints_all_frames[:, index, 0] = np.asarray(data_keypoint_preprocess_x)
        keypoints_all_frames[:, index, 1] = np.asarray(data_keypoint_preprocess_y)
        confidences_all_frames[:, index] = np.asarray(data_confidence_preprocess)

    cap.release()

    # Re-open the video to read frames again
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    os.makedirs(save_dir, exist_ok=True)
    while frame_idx < T:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = keypoints_all_frames[frame_idx]
        confidences = confidences_all_frames[frame_idx]

        # Convert normalized coordinates to image coordinates
        img_h, img_w = frame.shape[:2]
        x_coords = keypoints[:, 0] * img_w
        y_coords = keypoints[:, 1] * img_h

        # Prepare coords tensor for gen_gaussian_hmap_op
        coords = np.stack([x_coords, y_coords, confidences], axis=-1)[np.newaxis, :, :]  # Shape: (1, num_keypoints, 3)
        coords_tensor = torch.from_numpy(coords).float()

        # Generate heatmap
        hmap = gen_gaussian_hmap_op(
            coords_tensor, raw_size=(img_h, img_w), map_size=None,
            sigma=10, confidence=True, threshold=0.5)

        # Sum heatmaps over keypoints
        heatmap = hmap.sum(dim=1)[0]  # Shape: [H, W]

        # Normalize heatmap to 0-255
        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = (heatmap_np / np.max(heatmap_np) * 255).astype(np.uint8)

        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

        # Overlay heatmap on frame
        overlay = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)

        # Save the frame with heatmap overlay
        output_file = os.path.join(save_dir, f"heatmap_{frame_idx:05d}.jpg")
        cv2.imwrite(output_file, overlay)

        frame_idx += 1

    cap.release()


if __name__ == "__main__":
    video_path = "vsl/videos/01_Co-Hien_1-100_1-2-3_0108___center_device02_signer01_center_ord1_30.mp4"  # Replace with your video path
    save_directory = "vsl/heatmap"  # Replace with your desired save directory
    process_video(video_path, save_directory)
    print("Processing completed.")
