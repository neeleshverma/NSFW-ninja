import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml
from tqdm import tqdm

import torchvision
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

##### Import utils from parent directory
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import utils.classifier_utils as classifier_utils


resnet_weights = {'resnet18':ResNet18_Weights, 'resnet34':ResNet34_Weights, 'resnet50':ResNet50_Weights, 'resnet101':ResNet101_Weights}
devices = None


def getResnetModel(modelpath, modeltype):
    model = torchvision.models.__dict__[modeltype](weights=resnet_weights[modeltype].IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.to(devices[0])
    model = nn.DataParallel(model, device_ids=devices)

    checkpoint = torch.load(modelpath, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model


def getARTClassifier(model, config):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    return PyTorchClassifier(
        model=model,
        clip_values=(0, 255),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=2,
    )


def pgdAttack(config, num_images):
    test_dataset = classifier_utils.loadDataset(datadir = config['test_dir'], train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'])
    
    model = getResnetModel(config['model_path'], config['model'])
    classifier = getARTClassifier(model, config)

    normal_acc = 0
    adv_acc = 0
    asr = 0

    # PGD attack
    attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=0.2)


    for i, (image, target) in tqdm(enumerate(test_data_loader)):
        x_test = image.clone().numpy()
        target = target.type(torch.float)

        ### Normal output
        normal_outputs = classifier.predict(x_test)
        normal_outputs = torch.from_numpy(normal_outputs).squeeze()
        normal_preds = torch.round(torch.sigmoid(normal_outputs))
        normal_acc += torch.sum(normal_preds == target)

        ### Adversarial generation and output
        x_test_adv = attack.generate(x=x_test)
        adv_outputs = classifier.predict(x_test_adv)
        adv_outputs = torch.from_numpy(adv_outputs).squeeze()
        adv_preds = torch.round(torch.sigmoid(adv_outputs))
        adv_acc += torch.sum(adv_preds == target)

        print(normal_preds.shape)
        print(adv_preds.shape)
        print(type(normal_acc))
        print(type(adv_acc))

        asr += torch.sum(normal_preds != adv_preds)

    print("Normal acc : {}".format(normal_acc * 100.0 / (test_data_loader.dataset)))
    print("Adv acc : {}".format(adv_acc * 100.0 / (test_data_loader.dataset)))
    print("Attack Success Rate : {}".format(asr * 100.0 / (test_data_loader.dataset)))



if __name__ == "__main__":
    with open('configs/resnet_config.yaml') as f:
        config = yaml.safe_load(f)

    config['model_path'] = config['model_path'].format(model=config['model'])
    config['checkpoints'] = config['checkpoints'].format(model=config['model'])

    devices = [torch.device(f"cuda:{id}") for id in config['gpus']]
    num_images = 1000
    pgdAttack(config, num_images)