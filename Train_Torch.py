"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torchvision
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from CV_Models import end_to_end_Net
from Src.CV_Functions import Complex2foldloss, Complex2foldloss_Coh
from Src.Dataset_Preperation_Torch import BasicDataset

torch.autograd.set_detect_anomaly(True)

dir_checkpoint = 'checkpoints/'

net_params = {"epochs": 10,
              "batch_size": 10,
              "learning_rate": 0.001,
              "val_percentage": 0,
              "img_scale": 0.5}

def train_net(net,
              device,
              epochs=net_params["epochs"],
              batch_size=net_params["batch_size"],
              lr=net_params["learning_rate"],
              val_percent=net_params["val_percentage"],
              save_cp=True,
              img_scale=net_params["img_scale"]):

    ############ Datasets

    dataset = np.load("Path to the data")

    # dataset = np.expand_dims(dataset, axis=1)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    labels = np.load("patch to the labels")
    ######### 3-class label classification
    # for i in range(len(labels_temp)):
    #     if labels_temp[i] == 1 or labels_temp[i] == 2:
    #         labels[i] = 0
    #     elif labels_temp[i] == 3 or labels_temp[i] == 4 or labels_temp[i] == 5 or labels_temp[i] == 6:
    #         labels[i] = 1
    #     elif labels_temp[i] == 7:
    #         labels[i] = 2

    print("Labels and their counts:", np.unique(labels, return_counts=True))

    datadict = BasicDataset(imgs=dataset, labels=labels, scale=img_scale, normal=False)

    train_loader = DataLoader(datadict, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    # global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = Complex2foldloss_Coh(alpha=0.5)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                imgs = batch["image"]
                label = batch["label"]

                target = imgs[:, 0:2, :, :]
                assert imgs.shape[1] == net.n_in_channels, \
                    f'Network has been defined with {net.n_in_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # imgs = imgs.to(device=device, dtype=torch.float32)
                imgs = imgs.to(device=device, dtype=torch.complex64)
                label = label.to(device=device, dtype=torch.long)
                target_type = torch.complex64
                target = target.to(device=device, dtype=target_type)

                rec_imgs, classified_pred = net(imgs)
                loss, reconstroction_loss_temp, classification_loss_temp = criterion(reconstroction_predicted=rec_imgs, reconstroction_target=target, classification_predicted=classified_pred, classification_target=label)
                epoch_loss += loss.item()

                # writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

        torch.save(net.state_dict(), 'path to save model_epoch{}.pth'.format(epoch))

                # global_step += 1
                # if global_step % (n_train // (1 * batch_size)) == 0:
                #     for tag, value in net.named_parameters():
                #         tag = tag.replace('.', '/')
                #         writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                #         writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                #     val_score = eval_net(net, val_loader, device)
                #     scheduler.step(val_score)
                #     writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                #
                #     if net.n_classes > 1:
                #         logging.info('Validation cross entropy: {}'.format(val_score))
                #         writer.add_scalar('Loss/test', val_score, global_step)
                #     else:
                #         logging.info('Validation Dice Coeff: {}'.format(val_score))
                #         writer.add_scalar('Dice/test', val_score, global_step)
                #
                #     writer.add_images('images', imgs, global_step)
                #     if net.n_classes == 1:
                #         writer.add_images('target', target, global_step)
                #         writer.add_images('reconstructed', torch.sigmoid(rec_imgs) > 0.5, global_step)
        #
        # if save_cp:
        #     try:
        #         os.mkdir(dir_checkpoint)
        #         logging.info('Created checkpoint directory')
        #     except OSError:
        #         pass
        #     torch.save(net.state_dict(),
        #                dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        #     logging.info(f'Checkpoint {epoch + 1} saved !')
    #
    # writer.close()
    # return np.squeeze(rec_losses), np.squeeze(class_losses)

def get_args(net_params):
    parser = argparse.ArgumentParser(description='Train the UNet on images and target images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args(['-e', str(net_params["epochs"]),
                              '-b', str(net_params["batch_size"]),
                              '-l', str(net_params["learning_rate"]),
                              '-s', str(net_params["img_scale"]),
                              '-v', str(100*net_params["val_percentage"])])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args(net_params=net_params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = end_to_end_Net(n_in_channels=8, n_out_channels=2, n_classes=7, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_in_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net, device=device, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr,
                  val_percent=args.val / 100, img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


    torch.save(net.state_dict(), 'path to save the trained model.pth')

