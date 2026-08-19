import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
import torch.optim.lr_scheduler as lr_scheduler
import yaml
import argparse
import os
from pathlib import Path

from Src.Custom_Datasets import custom_dataset
from CV_Models import get_net
from Src.Utils import setup_finetuning
from Src.CV_Losses import get_loss_function
from Trainer import train


# config_path = 'Config_SingleModel.yaml'

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # check the root to set up the paths
    if config['training']['root'] == 'Local':
        root = 'R:/'
    elif config['training']['root'] == 'Server':
        root = '/exports/csce/datastore/geos/users/rmohamm2/'

    # Setup paths
    save_path = Path(root) / config['training']['save_dir'] / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{config['description']}"
    # save_path = Path(config['training']['save_dir']) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{config['description']}_jobID-{job_id}"
    os.makedirs(save_path / 'Models', exist_ok=True)

    # Save config to the experiment directory
    with open(save_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Setup logging
    logger = logging.getLogger(config['description']+"-Training")
    logging.basicConfig(filename=save_path / 'training.log', level=logging.INFO)
    event_writer = SummaryWriter(save_path / 'tb_logs')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        logger.warning("----Device is CPU----")
    # load custom datasets
    train_dataset = custom_dataset(
        root=root,
        dataset_path=config['data']['train_data_path'],
        data_channels=config['data']['data_channels'],
        range_downsample_factor=config['data']['range_downsample_factor'],
        azimuth_downsample_factor=config['data']['azimuth_downsample_factor'],
        logger = logger,
        data_normalization=config['data']['dataset_normalization'],
        data_stats_dir=config['data']['data_stats_dir'],
        clipping_threshold=config['data']['clipping_threshold'],
        transform=None
    )
    eval_dataset = custom_dataset(
        root=root,
        dataset_path=config['data']['eval_data_path'],
        data_channels=config['data']['data_channels'],
        range_downsample_factor=config['data']['range_downsample_factor'],
        azimuth_downsample_factor=config['data']['azimuth_downsample_factor'],
        logger = logger,
        data_normalization=config['data']['dataset_normalization'],
        data_stats_dir=config['data']['data_stats_dir'],
        clipping_threshold=config['data']['clipping_threshold'],
        transform=None
    )
    # Create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config['training']['train_batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=config['training']['num_workers'],
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=config['training']['eval_batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=config['training']['num_workers'],
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # Initialize model
    net = get_net(model_config=config['model'])
    net = net.to(device)

    # Load pretrained weights if specified
    if config['model']['pretrained']:
        checkpoint = torch.load(Path(root) / Path(config['model']['checkpoint_path']), map_location=device, weights_only=False)
        # checkpoint = torch.load(config['model']['checkpoint_path'], map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        total_step = checkpoint['total_step']
        best_evalloss = checkpoint['eval_loss']
        start_epoch = checkpoint['epoch']
    else:
        total_step = 0
        start_epoch = 0
        best_evalloss = float('inf')

    # net = torch.compile(net, mode="reduce-overhead")
    net = setup_finetuning(net=net, finetuning_strategy=config['model']['finetuning_strategy'])

    # Setup training
    if config['training']['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            net.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            net.parameters(),
            lr=config['training']['learning_rate'],
            betas=config['training']['betas'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        logger.warning("Optimizer is not defined!!!!")

    if config['training']['use_lr_scheduler']:
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['lr_scheduler_step'],
            gamma=config['training']['lr_scheduler_gamma']
        )
    else:
        scheduler = None

    criterion = get_loss_function(loss_function_name=config['training']['loss_function'])

    for epoch in range(start_epoch, config['training']['total_epochs']):
        total_step, best_evalloss = train(
            net=net,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            epoch=epoch,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            eval_freq=config['training']['eval_freq'],
            total_step=total_step,
            total_epochs=config['training']['total_epochs'],
            best_evalloss=best_evalloss,
            save_path=save_path,
            event_writer=event_writer,
            logger=logger,
            clip_max_norm=config['training']['clip_max_norm']
        )

    # Save final model
    checkpoint_dict = {
        'epoch': epoch,
        'total_step': total_step,
        'state_dict': net.state_dict(),
        'eval_loss': best_evalloss
    }
    torch.save(checkpoint_dict, save_path / 'Models/Model_Final.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    args = parser.parse_args()
    main(args.config)