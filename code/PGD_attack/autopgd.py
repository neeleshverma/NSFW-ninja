from autoattack import AutoAttack
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import yaml
import sys
from tqdm import tqdm
from art.estimators.classification import PyTorchClassifier
import torchvision
import torch.nn as nn
import imageio

devices = None

from art.attacks.evasion import ProjectedGradientDescentPyTorch

def getResnetModel(modelpath, modeltype):
    model = torchvision.models.__dict__[modeltype](weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(devices[0])
    model = nn.DataParallel(model, device_ids=devices)

    checkpoint = torch.load(modelpath, map_location='cpu')
    # print(checkpoint)
    # state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    # model.load_state_dict(state_dict)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model

def getARTClassifier(model, config):
    mean = config.get('mean', 0.0)
    std = config.get('std', 1.0)
    preprocessing = (mean, std)
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=torch.nn.BCEWithLogitsLoss(),
        input_shape=(3, 224, 224),
        nb_classes=2,
        preprocessing=preprocessing,
    )
    return classifier

if __name__ == "__main__":
    with open('/home/neelesh/NSFW-ninja/code/configs/resnet_config.yaml') as f:
        config = yaml.safe_load(f)
    devices = [torch.device(f"cuda:{id}") for id in config['gpus']]
    print(devices)
    # config['model'] = sys.argv[1]

    config['model_path'] = config['model_path'].format(model=config['model'])
    config['checkpoints'] = config['checkpoints'].format(model=config['model'])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.get('mean', 0.0), std=config.get('std', 1.0))    
        ])

    test_dataset = datasets.ImageFolder(root=config['test_dir'], transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'])

    model = getResnetModel(config['model_path'], config['model'])
    # classifier = getARTClassifier(model, config)

    normal_acc = 0
    adv_acc = 0
    asr = 0
    total_images = 0

    # AutoAttack
    attack = AutoAttack(model, norm='Linf', eps=8/255, version='standard', verbose=True)
    l_inf_max = 0

    for i, (image, target) in tqdm(enumerate(test_data_loader)):
        print("Batch : ", i)
        print("Total images : ", total_images)
        total_images += target.shape[0]
        print("Total images : ", total_images)

        x_test = image.clone().numpy()
        # target = target.type(torch.float)
        # print("Target : ", target)

        ### Normal output
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_test_tensor = torch.from_numpy(x_test).to(device)
        x_test_tensor.requires_grad_(True)
        normal_outputs = model(x_test_tensor)
        # print(normal_outputs, normal_outputs.shape)
        # print(target, target.shape)
        normal_outputs = normal_outputs.squeeze()        
        # normal_preds = torch.round(torch.sigmoid(normal_outputs))
        _, normal_preds = torch.max(normal_outputs, dim=1)
        target = target.to(device) 
        # print(normal_preds, target)
        print(normal_preds.shape, target.shape)
        normal_acc += torch.sum(normal_preds == target)
        print("Normal acc : ", normal_acc * 100.0 / total_images)

        ### Adversarial generation and output
        x_test_tensor = torch.from_numpy(x_test).to(device)
        x_test_tensor.requires_grad_(True)

        attack.set_version('rand')
        x_test_adv = attack.run_standard_evaluation(x_test_tensor, target,bs=128)        
        # save the adversarial images
        # for j in range(x_test_adv.shape[0]):
            # img = x_test_adv[j].cpu().numpy().transpose(1, 2, 0)
            # img = (img * 255).astype(np.uint8)
            # imageio.imsave('/home/neelesh/NSFW-ninja/adv_images/'+str(i)+'_'+str(j)+'.png', img)
        adv_outputs = model(x_test_adv)
        adv_outputs = adv_outputs.squeeze()
        # adv_preds = torch.round(torch.sigmoid(adv_outputs))
        _, adv_preds = torch.max(adv_outputs, dim=1)

        adv_acc += torch.sum(adv_preds == target)
        print("Adversarial acc : ", adv_acc * 100.0 / total_images)

        x_test_adv_np = x_test_adv.cpu().numpy()  # Convert to numpy array

        
        l_inf_max = max(l_inf_max, np.max(np.abs(x_test - x_test_adv_np)))        
        print("L-inf norm : ", l_inf_max)

        asr += torch.sum(normal_preds != adv_preds)

    ### Logging
    log = "Model : {}\n".format(config['model'])
    log += "AutoAttack Lib\n"
    log += "Normal acc : {}\n".format(normal_acc * 100.0 /
                                        total_images)
    log += "Adversarial acc : {}\n".format(adv_acc * 100.0 /
                                           total_images)
    log += "Attack Success Rate : {}\n".format(asr * 100.0 /
                                               total_images)
    log += "L-inf Norm Max : {}\n".format(l_inf_max)
    log += "--------------------------------------------------------------\n\n"
    log_file = open(config['logfile'], "a")
    log_file.write(log)

    log_file.close()
