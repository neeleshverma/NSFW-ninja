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

devices = None

resnet_weights = {'resnet18':ResNet18_Weights, 'resnet34':ResNet34_Weights, 'resnet50':ResNet50_Weights, 'resnet101':ResNet101_Weights}

def test(args):
    test_dataset = classifier_utils.loadDataset(datadir = args['test_dir'], train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args['batch_size'],
        shuffle=False, num_workers=args['num_workers'], pin_memory=args['pin_memory'])

    # Architecture Loading
    model = torchvision.models.__dict__[args['model']](weights=resnet_weights[args['model']].IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.to(devices[0])
    model = nn.DataParallel(model, args['gpus'])

    # Best model file from saved models
    best_model_file = args['model_path']

    # Model Loading
    checkpoint = torch.load(best_model_file, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Model Loading completed

    correct_preds = 0
    test_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for i, (image, target) in enumerate(test_data_loader):
            image, target = image.to(devices[0]), target.to(devices[0])
            target = target.type(torch.float)
            
            output = model(image)
            output = output.squeeze()

            loss = criterion(output, target)

            test_loss += loss.item()
            y_pred = torch.round(torch.sigmoid(output))
            correct_preds += torch.sum(y_pred == target).cpu()
    
    acc = correct_preds / len(test_data_loader.dataset)
    print("Test Accuracy : {}".format(acc))
    print("")

if __name__ == "__main__":
    with open('configs/resnet_config.yaml') as f:
        config = yaml.safe_load(f)

    config['model_path'] = config['model_path'].format(model=config['model'])
    config['checkpoints'] = config['checkpoints'].format(model=config['model'])

    devices = [torch.device(f"cuda:{id}") for id in config['gpus']]

    test(config)
