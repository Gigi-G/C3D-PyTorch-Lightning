import pytest
import torch
import sys
import os

from torch.utils.data import DataLoader

def test_model():
    with pytest.raises(SystemExit):
        sys.path.append(os.getcwd())
        sys.path.append(os.path.join(os.getcwd(), "modules"))
        from modules.C3D import C3D
        inputs = torch.rand(1, 3, 16, 112, 112)
        net = C3D(num_classes=101)
        outputs = net.forward(inputs)
        assert outputs.size() == torch.Size([1, 101])
        raise SystemExit(1)
    
def test_pretrained_model():
    with pytest.raises(SystemExit):
        sys.path.append(os.getcwd())
        sys.path.append(os.path.join(os.getcwd(), "modules"))
        from modules.C3D import C3D
        inputs = torch.rand(1, 3, 16, 112, 112)
        net = C3D(num_classes=101, pretrained="./models/c3d-pretrained.pth")
        outputs = net.forward(inputs)
        assert outputs.size() == torch.Size([1, 101])
        raise SystemExit(1)
    
def test_dataset_ucf101():
    """ Test the dataset 

        It could take a while to preprocess the dataset.
    """
    with pytest.raises(SystemExit):
        sys.path.append(os.getcwd())
        sys.path.append(os.path.join(os.getcwd(), "modules"))
        from modules.dataloaders import VideoDataset
        if not os.path.exists('./dataset/UCF-101'):
            print("Dataset not found. Please download it first.")
            raise SystemExit(-1)
        train_data = VideoDataset(root_dir='./dataset/UCF-101', output_dir='./output', dataset='ucf101', split='train', clip_len=16, preprocess=True)
        assert len(train_data) == 2159
        train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)
        for i, sample in enumerate(train_loader):
            assert sample[0].size() == torch.Size([100, 3, 16, 112, 112])
            assert sample[1].size() == torch.Size([100])
            if i == 5:
                break
        raise SystemExit(1)