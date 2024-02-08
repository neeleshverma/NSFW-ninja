from __future__ import print_function
import datetime
import os
import time
import math

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

import utils
import utils.dataloader


def train_one_epoch(model, criterion, optimizer, data_loader, epoch, val_dataloader, classes, args):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    epoch_data_len = len(data_loader.dataset)
    print('Train data num: {}'.format(epoch_data_len))

    for i, (image, target) in enumerate(data_loader):
        batch_start = time.time()
        image, target = image.cuda(), target.cuda()
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(output, 1)

        loss_ = loss.item() * image.size(0) # this batch loss
        correct_ = torch.sum(preds == target.data) # this batch correct number

        running_loss += loss_
        running_corrects += correct_

        batch_end = time.time()
        if i % args['print_freq'] == 0 and i != 0:
            print('[TRAIN] Epoch: {}/{}, Batch: {}/{}, BatchAcc: {:.4f}, BatchLoss: {:.4f}, BatchTime: {:.4f}'.format(epoch,
                  args['epochs'], i, math.ceil(epoch_data_len/args['batch_size']), correct_.double()/image.size(0),
                  loss_/image.size(0), batch_end-batch_start))

        # Validation
        if i % args['eval_freq'] == 0 and i != 0:
            val_acc = validate(model, criterion, val_dataloader, epoch, i, args)
            model.train()
            # the first or best will be saved (based on validation accuracy)
            if len(g_val_accs) == 0 or val_acc > g_val_accs.get(max(g_val_accs, key=g_val_accs.get), 0.0):
                if args['checkpoints']:
                    torch.save({
                        'model': model.state_dict(),
                        'classes': classes,
                        'args': args},
                        os.path.join(args['checkpoints'], 'model_{}_{}.pth'.format(epoch, i)))
                    print('*** SAVE.DONE. VAL_BEST_INDEX: {}_{}, VAL_BEST_ACC: {} ***'.format(epoch, i, val_acc))
            g_val_accs[str(epoch)+'_'+str(i)] = val_acc
            k = max(g_val_accs, key=g_val_accs.get)
            print('val_best_index: [ {} ], val_best_acc: [ {} ]'.format(k, g_val_accs[k]))

    lr = optimizer.param_groups[0]["lr"]
    epoch_loss = running_loss / epoch_data_len
    epoch_acc = running_corrects.double() / epoch_data_len
    epoch_end = time.time()
    print('[Train@] Epoch: {}/{}, EpochAcc: {:.4f}, EpochLoss: {:.4f}, EpochTime: {:.4f}, lr: {}'.format(epoch,
          args['epochs'], epoch_acc, epoch_loss, epoch_end-epoch_start, lr))
    print()
    print()


def validate(model, criterion, data_loader, epoch, step, args):
    epoch_start = time.time()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    epoch_data_len = len(data_loader.dataset)
    print('Val data num: {}'.format(epoch_data_len))

    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            batch_start = time.time()
            image, target = image.cuda(), target.cuda()
            output = model(image)
            loss = criterion(output, target)

            _, preds = torch.max(output, 1)

            loss_ = loss.item() * image.size(0) # this batch loss
            correct_ = torch.sum(preds == target.data) # this batch correct number

            running_loss += loss_
            running_corrects += correct_

            batch_end = time.time()
            if i % args['print_freq'] == 0:
                print('[VAL] Epoch: {}/{}/{}, Batch: {}/{}, BatchAcc: {:.4f}, BatchLoss: {:.4f}, BatchTime: {:.4f}'.format(step,
                      epoch, args['epochs'], i, math.ceil(epoch_data_len/args['batch_size']), correct_.double()/image.size(0),
                      loss_/image.size(0), batch_end-batch_start))

        epoch_loss = running_loss / epoch_data_len
        epoch_acc = running_corrects.double() / epoch_data_len
        epoch_end = time.time()
        print('[Val@] Epoch: {}/{}, EpochAcc: {:.4f}, EpochLoss: {:.4f}, EpochTime: {:.4f}'.format(epoch,
              args['epochs'], epoch_acc, epoch_loss, epoch_end-epoch_start))
        print()
    return epoch_acc


def main(args):
    print("Loading data")
    traindir = args['train_dir']
    valdir = args['val_dir']
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()

    train_dataset = torchvision.datasets.ImageFolder(
             traindir,
             transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,]))

    print("Loading validation data")
    val_dataset = torchvision.datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,]))

    print("Creating data loaders")
    train_data_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args['batch_size'],
                    shuffle=True, num_workers=args['workers'], pin_memory=True)

    classes = train_data_loader.dataset.classes
    print(classes)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args['batch_size'],
        shuffle=False, num_workers=args['workers'], pin_memory=True)

    print("Creating model")
    model = torchvision.models.__dict__[args['model']](pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # support muti gpu
    model = nn.DataParallel(model, device_ids=[0,1])
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 80], gamma=args['lr_gamma'])


    if args['resume']:
        checkpoint = torch.load(args['resume'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


    print("Start training")
    start_time = time.time()
    for epoch in range(args['epochs']):
        train_one_epoch(model, criterion, optimizer, train_data_loader, epoch, val_dataloader, classes, args)
        lr_scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    with open('configs.yaml') as f:
        config = yaml.safe_load(f)


    if not os.path.exists(config['checkpoints']):
        os.mkdir(config['checkpoints'])

    g_val_accs = {}

    main(config)