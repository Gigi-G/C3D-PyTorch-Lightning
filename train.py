import os
import random
import click
from datetime import datetime

import torch
import numpy as np
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from os.path import join

from modules.dataloaders import VideoDataset
from modules.C3D import C3D
from modules.C3D import TrainC3D

@click.command()
@click.option('--dataset', default='ucf101', help='This is the dataset name.')
@click.option('--epochs', default=100, help='This is the number of epochs.')
@click.option('--test', is_flag=True, default=False, help='This is the test flag.')
@click.option('--snapshot_interval', default=50, help='This is the snapshot interval.')
@click.option('--batch_size', default=100, help='This is the batch size.')
@click.option('--lr', default=1e-3, help='This is the learning rate.')
@click.option('--num_workers', default=4, help='This is the number of workers.')
@click.option('--clip_len', default=16, help='This is the clip length.')
@click.option('--preprocess', default=False, help='This is the preprocess flag.')
@click.option('--pretrained', default="./models/c3d-pretrained.pth", help='This is the pretrained model path.')
@click.option('--root_dir', default="./dataset/UCF-101", help='This is the root directory of the dataset.')
@click.option('--output_dir', default="./output", help='This is the output directory.')
@click.option('--device', default="cuda", help='This is the device.')
@click.option('--seed', default=42, help='This is the seed.')
@click.option('--wandb_log', is_flag=True, default=False, help='This is the wandb flag.', type=bool)
@click.option('--checkpoint', default=None, help='This is the checkpoint path.')
def main(
    dataset:str, epochs:int, test:bool,
    snapshot_interval:int, batch_size:int, lr:float, num_workers:int,
    clip_len:int, preprocess:bool, pretrained:str, root_dir:str,
    output_dir:str, device:str, seed:int, wandb_log:bool, checkpoint:str
):
    # Check if dataset is supported
    if dataset != "ucf101":
        print ("Dataset not supported")
        raise NotImplementedError
    set_seed(seed)
    device = check_device(device)
    logger = None
    if wandb_log:
        logger = WandbLogger(project='c3d-pytorch-lightning', log_model=True)
    saveName = f"C3D - {dataset} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    save_folder = join(output_dir, saveName)
    os.makedirs(save_folder, exist_ok=True)
    print("Saving to: ", save_folder)
    model = C3D(num_classes=101, pretrained=pretrained)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = DataLoader(
        VideoDataset(root_dir=root_dir, output_dir=output_dir, dataset=dataset, split='train', clip_len=clip_len, preprocess=preprocess),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        VideoDataset(root_dir=root_dir, output_dir=output_dir, dataset=dataset, split='val', clip_len=clip_len, preprocess=preprocess),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        VideoDataset(root_dir=root_dir, output_dir=output_dir, dataset=dataset, split='test', clip_len=clip_len, preprocess=preprocess),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("Training started")
    train_C3D = TrainC3D(lr=lr, model=model, save_dir=save_folder, save_epochs=snapshot_interval)
    trainer = L.Trainer(max_epochs=epochs, accelerator="gpu", devices=1, logger=logger)
    trainer.fit(train_C3D, train_loader, val_loader, ckpt_path=checkpoint)
    if test and epochs > 0:
        print("Testing started")
        trainer.test(train_C3D, test_loader)
    elif test:
        print("Testing started")
        model = C3D(num_classes=101)
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
        train_C3D = TrainC3D(lr=lr, model=model, save_dir=save_folder, save_epochs=snapshot_interval)
        trainer.test(train_C3D, test_loader)
    print("Saving model")
    torch.save(model.state_dict(), join(save_folder, "c3d.pth"))
    
def set_seed(seed:int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def check_device(device:str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using {device} device")
    return device         

if __name__ == '__main__':
    main()