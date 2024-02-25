import utils.classifier_utils as classifier_utils

import time
import yaml
import os
import sys
from tqdm import tqdm

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import gc

# def extract_number(filename):
#     return int(filename.split('_')[1].split('.')[0])

with open('configs/inception_config.yaml') as f:
        config = yaml.safe_load(f)

model = "resnet101"
saved_models = "checkpoints"

config['checkpoints'] = config['checkpoints'].format(model=model)
config['plots'] = config['plots'].format(model=model)
config['model'] = model

devices = [torch.device(f"cuda:{id}") for id in [3,4]]

test_dataset = classifier_utils.loadDataset(datadir = config['test_dir'], train=False)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config['batch_size'],
    shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory'])

model_folder = os.path.join(saved_models, model)
model_files = os.listdir(model_folder)
model_files = sorted(model_files, key=extract_number)

acc_list = []

test_epochs_list = [5*i for i in range(1,21)]

for i in range(len(model_files)):
    model = torchvision.models.__dict__[config['model']](weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    model.to(devices[0])
    model = nn.DataParallel(model, device_ids=devices)

    checkpoint = torch.load(os.path.join(model_folder, model_files[i]), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    correct_preds = 0

    with torch.no_grad():
        for i, (image, target) in enumerate(tqdm(test_data_loader)): 
            image, target = image.to(devices[0]), target.to(devices[0])
            target = target.type(torch.float)
        
            output = model(image)
            output = output.squeeze()

            y_pred = torch.round(torch.sigmoid(output))
            correct_preds += torch.sum(y_pred == target).cpu()

    acc = correct_preds / len(test_data_loader.dataset)
    acc_list.append(acc.item())

    del model
    gc.collect()
    torch.cuda.empty_cache() 

# print(acc_list)
plt.plot(test_epochs_list, acc_list)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Test Accuracy vs Epochs")
plt.savefig(os.path.join(config['plots'], 'test.png'))
plt.close()

