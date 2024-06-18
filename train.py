import numpy as np
import math
import matplotlib.pyplot as plt
import random
import os
import torch
from path import Path
from source import model
from source import dataset
from source import utils
from source.args import parse_args
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer
import traceback
from sys import stdout

from myutils import print_train_time, plot_loss_curves

random.seed = 42

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)



def train(args):
    path = Path(args.root_dir)
    
    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
    classes = {folder: i for i, folder in enumerate(folders)};
    
    train_transforms = transforms.Compose([
        utils.PointSampler(1024),
        utils.Normalize(),
        utils.RandRotation_z(),
        utils.RandomNoise(),
        utils.ToTensor()
    ])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    pointnet = model.PointNet()
    pointnet.to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=args.lr)
    
    train_ds = dataset.PointCloudData(path, transform=train_transforms)
    valid_ds = dataset.PointCloudData(path, valid=True, folder='test', transform=train_transforms)
    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))
    print('Number of classes: ', len(train_ds.classes))
    
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=args.batch_size*2)
    
    try:
        os.mkdir(args.save_model_path)
    except OSError as error:
        print(error)
    
    # Empty lists to track loss and acc values
    results = []
    train_loss_values = []
    test_acc= []

    print('Start training')
    for epoch in range(args.epochs):
        train_time_start = timer()  # start training timer
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    train_loss_values.append(running_loss)
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    running_loss = 0.0

        # Calculate training time
        train_time_end = timer()
        print_train_time(
            start=train_time_start,
            end= train_time_end,
            device=device
        )
        # Add train_loss and batch_count to model checkpoint dictionary
        chkpoint_dict = {
            "train_loss": train_loss_values
        }
        results.append(chkpoint_dict)

        pointnet.eval()
        correct = total = 0
        
        # validation
        if valid_loader:
            with torch.no_grad():
                for data in valid_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)
            # Record test accuracy of model checkpoint
            test_acc.append(val_acc)
            results[epoch]['test_acc'] = test_acc


        # save the model
        checkpoint = Path(args.save_model_path)/'save_'+str(epoch)+'.pth'
        torch.save(pointnet.state_dict(), checkpoint)
        print('Model saved to ', checkpoint)

    return results
    
if __name__ == '__main__':
    args = parse_args()
    train_results = train(args)
    try:
        plot_loss_curves(train_results)
    except Exception as e:
            print(f"An unexpected exception occured of type {type(e)}")
            print(traceback.format_exc(limit=2))        