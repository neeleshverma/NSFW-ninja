import utils.classifier_utils as classifier_utils

import time
import yaml
import os
import sys

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000
devices = None

def train_epoch(model, criterion, optimizer, train_data_loader, current_epoch, args):
    model.train()
    training_loss = 0.0
    correct_preds = 0
    epoch_data_len = len(train_data_loader.dataset)

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

        y_pred = torch.round(torch.sigmoid(output))
        correct_preds += torch.sum(y_pred == target).cpu()
    
    acc = correct_preds.cpu() / epoch_data_len
    if current_epoch % args['print_freq'] == 0:
        print("Epoch : {}   Training Accuracy : {}".format(current_epoch, acc))
    return acc.item()


def save_model(model, optimizer, lr_scheduler, args, current_epoch):
    print("Saving the model at {}".format(args['checkpoints']))
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler},
        os.path.join(args['checkpoints'], 'model_{}.pth'.format(current_epoch)))
    print("")


def validate(model, criterion, val_data_loader, current_epoch, val_or_test="Validation"):
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
    print("Epoch : {}   {} Accuracy : {}".format(current_epoch, val_or_test, acc))
    return acc.item()


def train(args):
    dataset = classifier_utils.loadDataset(datadir = args['train_dir'], train=True)
    test_dataset = classifier_utils.loadDataset(datadir = args['test_dir'], train=False)

    train_dataset_size = int(0.8 * len(dataset))
    val_dataset_size = len(dataset) - train_dataset_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (train_dataset_size, val_dataset_size))

    train_data_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'],
        shuffle=True, num_workers=args['num_workers'], pin_memory=args['pin_memory'])
    
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args['batch_size'],
        shuffle=False, num_workers=args['num_workers'], pin_memory=args['pin_memory'])
    
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args['batch_size'],
        shuffle=False, num_workers=args['num_workers'], pin_memory=args['pin_memory'])

    model = torchvision.models.__dict__[args['model']](weights=None)
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
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    train_epochs_list = []
    test_epochs_list = []

    print("############# TRAINING ################")
    for epoch in range(1,args['epochs']+1):
        train_acc = train_epoch(model, criterion, optimizer, train_data_loader, epoch, args)
        train_acc_list.append(train_acc)
        train_epochs_list.append(epoch)
        lr_scheduler.step()
        if epoch % args['eval_freq'] == 0:
            test_epochs_list.append(epoch)
            val_acc = validate(model, criterion, val_data_loader, epoch)
            val_acc_list.append(val_acc)

            test_acc = validate(model, criterion, test_data_loader, epoch, val_or_test="Test")
            test_acc_list.append(test_acc)
            # if acc > val_acc:
            save_model(model, optimizer, lr_scheduler, args, epoch)
    
    classifier_utils.plot(train_epochs_list, train_acc_list, "Epochs", "Accuracy",
        "Training Accuracy vs Epochs", os.path.join(args['plots'], 'training.png'))
    classifier_utils.plot(test_epochs_list, val_acc_list, "Epochs", "Accuracy",
        "Validation Accuracy vs Epochs", os.path.join(args['plots'], 'validation.png'))
    classifier_utils.plot(test_epochs_list, test_acc_list, "Epochs", "Accuracy",
        "Test Accuracy vs Epochs", os.path.join(args['plots'], 'test.png'))


if __name__ == "__main__":
    with open('configs/resnet_config.yaml') as f:
        config = yaml.safe_load(f)

    model = sys.argv[1]

    config['checkpoints'] = config['checkpoints'].format(model=model)
    config['plots'] = config['plots'].format(model=model)
    config['model'] = model

    if not os.path.exists(config['checkpoints']):
        os.mkdir(config['checkpoints'])
    
    if not os.path.exists(config['plots']):
        os.makedirs(config['plots'])

    devices = [torch.device(f"cuda:{id}") for id in config['gpus']]
    train(config)