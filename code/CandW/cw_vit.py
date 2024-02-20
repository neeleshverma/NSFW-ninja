import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml
from tqdm import tqdm

import torchvision
import torchattacks

##### Import utils from parent directory
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import utils.classifier_utils as classifier_utils

devices = None

def getViTModel(modelpath, modeltype):
    model = torchvision.models.__dict__[modeltype](weights=None)
    num_ftrs = model.heads.head.in_features
    model.heads = nn.Linear(num_ftrs, 1)
    model.to(devices[0])
    model = nn.DataParallel(model, device_ids=devices)

    checkpoint = torch.load(modelpath, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model


def cw_l2_attack(model, images, labels, targeted=False, c=1e-2, kappa=0, max_iter=1000, learning_rate=0.01) :
    images = images.to(devices[0])     
    labels = labels.to(devices[0])

    def f(x):

        outputs = model(x)
        one_hot_labels = torch.eye(2)[labels.clone().cpu()].to(devices[0])

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        labels_cpy = labels.clone().eq(0).float()
        j = outputs * (1 - 2 * labels_cpy).view(-1,1)

        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        else :
            return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(images, requires_grad=True).to(devices[0])
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    prev = 1e10

    for step in range(max_iter) :

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %%' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1)

    return attack_images

def cwAttack(config, num_images):
    test_dataset = classifier_utils.loadDataset(datadir = config['test_dir'], train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'])
    
    model = getViTModel(config['model_path'], config['model'])

    normal_acc = 0
    adv_acc = 0
    total_images = 0
    asr = 0
    max_l2_norm = 0

    # C and W L2 attack
    # attack = torchattacks.CW(model, c=0.5, kappa=0, steps=50, lr=0.01)

    for i, (images, targets) in tqdm(enumerate(test_data_loader)):
        if total_images < num_images:
            total_images += targets.shape[0]

            adv_images = cw_l2_attack(model, images, targets, targeted=False, c=0.1)
            normal_images, targets = images.to(devices[0]), targets.to(devices[0])

            l2_norm = torch.max(torch.norm(normal_images - adv_images, p=2, dim=(1,2,3))).item()
            max_l2_norm = max(l2_norm, max_l2_norm)

            adv_outputs = model(adv_images)
            adv_outputs = adv_outputs.squeeze()
            adv_preds = torch.round(torch.sigmoid(adv_outputs))
            adv_acc += torch.sum(adv_preds == targets).cpu()

            normal_outputs = model(normal_images)
            normal_outputs = normal_outputs.squeeze()
            normal_preds = torch.round(torch.sigmoid(normal_outputs))
            normal_acc += torch.sum(normal_preds == targets).cpu()

            asr += torch.sum(normal_preds != adv_preds)


    ### Logging
    log = "Model : {}\n".format(config['model'])
    log += "Normal acc : {}\n".format(normal_acc * 100.0 / total_images)
    log += "Adversarial acc : {}\n".format(adv_acc * 100.0 / total_images)
    log += "Attack Success Rate : {}\n".format(asr * 100.0 / total_images)
    log += "Maximum L2 Norm : {}\n".format(max_l2_norm)
    log += "--------------------------------------------------------------"
    log_file = open(config['logfile'], "a")
    log_file.write(log)
    log_file.close()



if __name__ == "__main__":
    with open('configs/vit_config.yaml') as f:
        config = yaml.safe_load(f)

    config['model_path'] = config['model_path'].format(model=config['model'])
    config['checkpoints'] = config['checkpoints'].format(model=config['model'])

    devices = [torch.device(f"cuda:{id}") for id in config['gpus']]
    num_images = 1000
    cwAttack(config, num_images)