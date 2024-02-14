import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml
from tqdm import tqdm

import torchvision
from torchvision.models import Inception_V3_Weights
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

##### Import utils from parent directory
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import utils.classifier_utils as classifier_utils

devices = None

def getInceptionModel(modelpath, modeltype):
    model = torchvision.models.__dict__[modeltype](weights=None)
    num_ftrs = model.fc.in_features
    classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 1)
    )

    model.fc = classifier
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
    test_dataset = classifier_utils.loadInceptionDataset(datadir = config['test_dir'], train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'])
    
    model = getInceptionModel(config['model_path'], config['model'])
    classifier = getARTClassifier(model, config)

    normal_acc = 0
    adv_acc = 0
    asr = 0
    total_images = 0

    # PGD attack
    attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=8/255, norm="inf", batch_size=128, eps_step=2/255)
    l_inf_max = 0

    for i, (image, target) in tqdm(enumerate(test_data_loader)):
        if total_images < num_images:
            total_images += target.shape[0]
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

            l_inf_max = max(l_inf_max, np.max(np.abs(x_test - x_test_adv)))
            print("L-inf norm : ", l_inf_max)

            asr += torch.sum(normal_preds != adv_preds)

    ### Logging
    log = "Model : {}\n".format(config['model'])
    log += "Normal acc : {}\n".format(normal_acc * 100.0 / total_images)
    log += "Adversarial acc : {}\n".format(adv_acc * 100.0 / total_images)
    log += "Attack Success Rate : {}\n".format(asr * 100.0 / total_images)
    log += "L-inf Norm Max : {}\n".format(l_inf_max)
    log += "--------------------------------------------------------------"
    log_file = open(config['logfile'], "a")
    log_file.write(log)
    log_file.close()



if __name__ == "__main__":
    with open('configs/inception_config.yaml') as f:
        config = yaml.safe_load(f)

    config['model_path'] = config['model_path'].format(model=config['model'])
    config['checkpoints'] = config['checkpoints'].format(model=config['model'])

    devices = [torch.device(f"cuda:{id}") for id in config['gpus']]
    num_images = 1000
    pgdAttack(config, num_images)