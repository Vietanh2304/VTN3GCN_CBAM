import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
from einops import rearrange
from augumentation import Rotate,Compose
from torch.utils.data import random_split
import os
import pandas as pd
import math

# Class read npy and pickle file to make data and label in couple
class FeederINCLUDE(Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
    """
    def __init__(self, data_path: Path, label_path: Path, transform = None):
        super(FeederINCLUDE, self).__init__
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.load_data()
    
    def load_data(self):
        # data: N C V T M
        # Load label with numpy
        self.label = np.load(self.label_path)
        # load data
        self.data = np.load(self.data_path)     
        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __getitem__(self, index):
        """
        Input shape (N, C, V, T, M)
        N : batch size
        C : numbers of features
        V : numbers of joints (as nodes)
        T : numbers of frames
        M : numbers of people (should delete)
        
        Output shape (C, V, T, M)
        C : numbers of features
        V : numbers of joints (as nodes)
        T : numbers of frames
        label : label of videos
        """
        data_numpy = torch.tensor(self.data[index]).float()
        # Delete one dimension
        # data_numpy = data_numpy[:, :, :2]
        # data_numpy = rearrange(data_numpy, ' t v c 1 -> c t v 1')
        label = self.label[index]
        p = random.random()
        if self.transform and p > 0.5: 
            data_numpy, label = self.transform(data_numpy, label)
        return data_numpy, label
    
    def __len__(self):
        return len(self.label)

class FeederCustomV2(Dataset):
    """Feeder for skeleton-based action recognition.

    Arguments:
        base_url (str): Base directory of the dataset.
        split (str): Dataset split, one of 'train', 'val', or 'test'.
        num_output_frames (int): Number of frames to load per sample.
        label_folder (str): Folder name where label CSV files are stored.
        data_type (str): Type of data, e.g., 'keypoints'.
        train_labels (pd.DataFrame, optional): DataFrame of labels for custom splits.
    """
    def __init__(self, base_url, split, num_output_frames=80, label_folder='label1-200/label/labelCenterWithOrd1/labelCenterWithout29_ord1_4316_792_791', data_type='labels', train_labels=None):
        super(FeederCustomV2, self).__init__()
        if train_labels is None:
            # Load labels from the CSV file
            label_path = os.path.join(base_url, f"{label_folder}/{split}_{data_type}.csv")
            print("Label Path:", label_path)
            self.labels = pd.read_csv(label_path)
        else:
            print("Using provided train_labels")
            self.labels = train_labels

        print(f"{split} set size: {len(self.labels)}")
        self.split = split
        self.is_train = (split == 'train')
        self.base_url = base_url
        self.num_output_frames = num_output_frames
        self.transform = self.build_transform(split)
    
    def build_transform(self, split):
        # No transformations are applied to keypoints data in this example
        if split == 'train':
            transform = Compose([
                        Rotate(15, 80, 25, (0.5, 0.5))
                        ])
        else:
            transform = None
        return transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get data sample at index idx
        data_row = self.labels.iloc[idx].values
        name, label = data_row[0], data_row[1]

        # Read keypoints data (handflow)
        handflow = self.read_handflow(name)
        
        return handflow, torch.tensor(label)
    
    def read_handflow(self, name):
        # Read handflow data for the given sample name
        handflow_clip = []

        # Number of frames to load
        num_frames = self.num_output_frames

        for frame_index in range(num_frames):
            # Construct the path to the keypoints file
            frame_index_handflow = frame_index
            full_path = os.path.join(self.base_url, 'hand_keypoints', name.replace(".mp4", ""),
                                     f'hand_kp_{frame_index_handflow:05d}.npy')

            # Handle missing files by backtracking to previous frames
            while not os.path.isfile(full_path) and frame_index_handflow > 0:
                frame_index_handflow -= 1
                full_path = os.path.join(self.base_url, 'hand_keypoints', name.replace(".mp4", ""),
                                         f'hand_kp_{frame_index_handflow:05d}.npy')

            if os.path.isfile(full_path):
                # Load the keypoints data
                value = np.load(full_path)
                handflow_frame = value
            else:
                # If no handflow data is found, initialize with zeros
                handflow_frame = np.zeros((46, 2))

            # Apply transformations to handflow data
            handflow_frame = self.transform_handflow(handflow_frame)
            handflow_clip.append(handflow_frame)

        # Stack handflow frames into a tensor along the time dimension
        handflow = torch.stack(handflow_clip, dim=1)  # shape: [C, T, V]

        # Add the M dimension (number of persons), which is 1 in this case
        handflow = handflow.unsqueeze(-1)  # shape: [C, T, V, M]

        return handflow

    def transform_handflow(self, handflow):
        # Convert to a PyTorch tensor and transpose to get [C, V]
        handflow_tensor = torch.tensor(handflow, dtype=torch.float32).transpose(0, 1)
        return handflow_tensor


if __name__ == '__main__':
    # file, label = np.load("wsl100_train_data_preprocess.npy"), np.load("wsl100_train_label_preprocess.npy")
    # print(file.shape, label.shape)
    # data = FeederINCLUDE(data_path=f"/home/ibmelab/Documents/GG/VSLRecognition/vsl/AAGCN/vsl199_train_data_preprocess.npy", 
    #                      label_path=f"/home/ibmelab/Documents/GG/VSLRecognition/vsl/AAGCN/train_label_preprocess.npy",
    #                         transform=None)
    data = FeederCustomV2('/home/ibmelab/Documents/GG/VSLRecognition/vsl','train')
    # print(data.N, data.C, data.T, data.V, data.M)
    # print(data.data.shape)
    # print(data.__len__())

    for idx in range(len(data)):
        handflow, label = data[idx]
        print(f"Sample {idx}: Handflow shape: {handflow.shape}, Label: {label}")
        if idx == 5:  # Limit to first 5 samples
            break
