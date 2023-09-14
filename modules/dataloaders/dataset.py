import os
import torch
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

class VideoDataset(Dataset):
    """A dataset for a folder of videos. The structure is assumed to be:
    directory/[train/val/test]/[class]/[video files].
    """
    
    def __init__(self, root_dir:str, output_dir:str, dataset:str, split:str='train', clip_len:int=16, preprocess:bool=False) -> None:
        """Construct the VideoDataset object.
        
            Args:
                root_dir: The root directory of the dataset.
                output_dir: The output directory of the split dataset.
                dataset: The dataset name.
                split: The split name.
                clip_len: The length of the clips to be extracted from the videos.
                preprocess: Whether to preprocess the videos or not.
        """
        self.root_dir = root_dir
        self.split = split
        self.clip_len = clip_len
        self.output_dir = output_dir
        folder = os.path.join(output_dir, split)
        
        # Values tooked from the paper
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112
        
        # Check if the dataset exists.
        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')
        
        # Check if the dataset has been preprocessed.
        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing the dataset...')
            self.preprocess()
        
        # Obtain the list of the videos.
        self.fnames = []
        self.labels = []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                self.labels.append(label)
        
        assert len(self.fnames) == len(self.labels)
        print(f'Number of {split} videos: {len(self.fnames)}')
        
        self.label2index = {label: index for index, label in enumerate(sorted(set(self.labels)))}
        self.label_array = np.array([self.label2index[label] for label in self.labels], dtype=int)
        
        if dataset == 'ucf101':
            if not os.path.exists(os.path.join(output_dir, 'ucf_labels.txt')):
                with open(os.path.join(output_dir, 'ucf_labels.txt'), 'w') as f:
                    for id, label in enumerate(sorted(set(self.label2index))):
                        f.write(f'{id+1} {label}\n')
        else:
            print('Dataset not supported. Contribute to the repo if you want to add it.')
            raise NotImplementedError
    
    def __len__(self) -> int:
        """Return the number of clips.
        
            Returns:
                The number of clips.
        """
        return len(self.fnames)
    
    def __getitem__(self, index:int) -> torch.Tensor:
        """Get a clip from the dataset.

            Args:
                index: The index of the clip.
                
            Returns:
                clip: A torch tensor containing the clip.
        """
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])
        if self.split == 'train':
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)
    
    def crop(self, buffer:np.ndarray, clip_len:int, crop_size:int) -> np.ndarray:
        """Crop the given video at a random location and time.
        
            Args:
                buffer: The video to be cropped.
                clip_len: The length of the clip to be extracted.
                crop_size: The size of the crop.
                
            Returns:
                buffer: The cropped video.
        """
        # Randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)
        # Crop the video temporal and spatially
        buffer = buffer[time_index:time_index + clip_len,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size, :] 
        return buffer
    
    def randomflip(self, buffer:np.ndarray) -> np.ndarray:
        """Randomly flip the video frames horizontally.
        
            Args:
                buffer: The video to be flipped.
                
            Returns:
                buffer: The flipped video.
        """
        # Randomly flip the video frames horizontally
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer
    
    def normalize(self, buffer:np.ndarray) -> np.ndarray:
        """Normalize the video frames.
        
            Args:
                buffer: The video to be normalized.
                
            Returns:
                buffer: The normalized video.
        """
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer
    
    def to_tensor(self, buffer:np.ndarray) -> np.ndarray:
        """Convert the video frames to a torch tensor.
        
            Args:
                buffer: The video to be converted.
                
            Returns:
                buffer: The converted video.
        """
        return buffer.transpose((3, 0, 1, 2))
    
    def check_integrity(self) -> bool:
        """Check if the dataset directory exists."""
        return os.path.exists(self.root_dir)
    
    def check_preprocess(self) -> bool:
        """Check if the dataset has been preprocessed."""
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False
        
        print('Checking the preprocessed dataset...')
        for i, video_class in tqdm(enumerate(os.listdir(os.path.join(self.output_dir, 'train')))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                            sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break
            if i == 10:
                break
            
        return True
    
    def process_video(self, video:str, action_name:str, save_dir:str) -> None:
        """Extract frames from a video using opencv.
        
            Args:
                video: The video to be processed.
                action_name: The name of the action. (Path of the action folder)
                save_dir: The directory where to save the frames.
        """
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))
            
        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))
        
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        extract_frequency = 4
        if frame_count // extract_frequency <= self.clip_len:
            extract_frequency -= 1
            if frame_count // extract_frequency <= self.clip_len:
                extract_frequency -= 1
                if frame_count // extract_frequency <= self.clip_len:
                    extract_frequency -= 1
        
        count = 0
        i = 0
        retaining = True
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue
            if count % extract_frequency == 0:
                if frame_height != self.resize_height or frame_width != self.resize_width:
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(os.path.join(save_dir, video_filename, f'0000{i}.jpg'), frame)
                i += 1
            count += 1
            
        capture.release()
    
    def preprocess(self) -> None:
        """ Preprocess the dataset."""
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'val'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'test'), exist_ok=True)
        
        # Split the dataset into train, val and test
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]
            
            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)
            
            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)
            
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            print(f'Preprocessing {file}...')
            for video_train, video_val, video_test in tqdm(zip(train, val, test)):
                self.process_video(video_train, file, train_dir)
                self.process_video(video_val, file, val_dir)
                self.process_video(video_test, file, test_dir)
        
        print('Preprocessing finished.')
    
    def load_frames(self, file_dir:str) -> np.ndarray:
        """Load frames from a video using opencv.
        
            Args:
                file_dir: The directory of the video.
                
            Returns:
                frames: A numpy array containing the frames of the video.
        """
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float32)
            buffer[i] = frame
        return buffer
    
    