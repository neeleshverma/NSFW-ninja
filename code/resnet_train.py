import utils.classifier_utils as classifier_utils

import time
import yaml

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
import torch.nn.functional as F

import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

devices = None

resnet_weights = {'resnet18':ResNet18_Weights, 'resnet34':ResNet34_Weights, 'resnet50':ResNet50_Weights, 'resnet101':ResNet101_Weights}


def train_epoch(model, criterion, optimizer, train_data_loader, current_epoch, args):
    model.train()
    training_loss = 0.0
    correct_preds = 0
    epoch_data_len = len(train_data_loader.dataset)
    # print('Total Training Data: {}'.format(epoch_data_len))

    for i, (image, target) in enumerate(train_data_loader):
        image, target = image.to(devices[0]), target.to(devices[0])
        target = target.type(torch.float)
        
        output = model(image)
        output = output.squeeze()

        loss = criterion(output, target)

        training_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # acc = classifier_utils.accuracy(target, output)
        y_pred = torch.round(torch.sigmoid(output))
        correct_preds += torch.sum(y_pred == target).cpu()
        # print(acc)
    
    acc = correct_preds.cpu() / epoch_data_len
    if current_epoch % args['print_freq'] == 0:
        print("Epoch : {}   Training Accuracy : {}".format(current_epoch, acc))
        # print("")


def save_model(model, optimizer, lr_scheduler, args, current_epoch):
    print("Saving the model at {}".format(args['checkpoints']))
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler},
        os.path.join(args['checkpoints'], 'model_{}.pth'.format(current_epoch)))
    print("")


def validate(model, criterion, val_data_loader, current_epoch):
    model.eval()
    validation_loss = 0.0
    correct_preds = 0
    with torch.no_grad():
        for i, (image, target) in enumerate(val_data_loader): 
            image, target = image.to(devices[0]), target.to(devices[0])
            target = target.type(torch.float)
            
            output = model(image)
            output = output.squeeze()

            loss = criterion(output, target)

            validation_loss += loss.item()
            y_pred = torch.round(torch.sigmoid(output))
            correct_preds += torch.sum(y_pred == target).cpu()
    
    acc = correct_preds / len(val_data_loader.dataset)
    print("Epoch : {}   Validation Accuracy : {}".format(current_epoch, acc))
    print("")
    return acc


def train(args):
    dataset = classifier_utils.loadDataset(datadir = args['train_dir'], train=True)
    train_dataset_size = int(0.7 * len(dataset))
    val_dataset_size = len(dataset) - train_dataset_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (train_dataset_size, val_dataset_size))

    train_data_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'],
        shuffle=True, num_workers=args['num_workers'], pin_memory=args['pin_memory'])
    
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args['batch_size'],
        shuffle=False, num_workers=args['num_workers'], pin_memory=args['pin_memory'])


    model = torchvision.models.__dict__[args['model']](weights=resnet_weights[args['model']].IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    model.to(devices[0])
    model = nn.DataParallel(model, device_ids=devices)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['optim_milestones'], gamma=args['lr_gamma'])


    if args['resume']:
        checkpoint = torch.load(args['model_path'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # Training
    val_acc = 0
    print("############# TRAINING ################")
    for epoch in range(args['epochs']):
        train_epoch(model, criterion, optimizer, train_data_loader, epoch, args)
        lr_scheduler.step()
        if epoch % args['eval_freq'] == 0:
            acc = validate(model, criterion, val_data_loader, epoch)
            if acc > val_acc:
                save_model(model, optimizer, lr_scheduler, args, epoch)

    


if __name__ == "__main__":
    with open('configs/resnet_config.yaml') as f:
        config = yaml.safe_load(f)

    config['model_path'] = config['model_path'].format(model=config['model'])
    config['checkpoints'] = config['checkpoints'].format(model=config['model'])

    if not os.path.exists(config['checkpoints']):
        os.mkdir(config['checkpoints'])

    devices = [torch.device(f"cuda:{id}") for id in config['gpus']]

    train(config)