import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, Inception_V3_Weights
import matplotlib.pyplot as plt

def loadDataset(datadir=None, train=False):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return torchvision.datasets.ImageFolder(
                datadir,
                transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(30),
                        transforms.ToTensor()]))
    else:
        return torchvision.datasets.ImageFolder(
                datadir,
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()]))


def loadInceptionDataset(datadir=None, train=False):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return torchvision.datasets.ImageFolder(
                datadir,
                transforms.Compose([
                        transforms.Resize((342, 342)),
                        transforms.RandomCrop(299),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(30),
                        transforms.ToTensor()]))
    else:
        return torchvision.datasets.ImageFolder(
                datadir,
                transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.ToTensor()]))


def accuracy(y, y_pred):
    y_pred = torch.round(torch.sigmoid(y_pred))
    return torch.sum(y_pred == y) / y.shape[0]


def loadModel(args):
    devices = [torch.device(f"cuda:{id}") for id in args['gpus']]
    args['model_path'] = args['model_path'].format(model=args['model'])
    
    if args['base'] == 'resnet':

        # resnet_weights = {'resnet18':ResNet18_Weights, 'resnet34':ResNet34_Weights, 'resnet50':ResNet50_Weights}
        resnet_weights = {'resnet18':ResNet18_Weights, 'resnet34':ResNet34_Weights, 'resnet50':ResNet50_Weights, 'resnet101':ResNet101_Weights}
        
        model = torchvision.models.__dict__[args['model']](weights=resnet_weights[args['model']].IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        model.to(devices[0])
        model = nn.DataParallel(model, args['gpus'])

        best_model_file = args['model_path']

        checkpoint = torch.load(best_model_file, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        model.eval()

    elif args['base'] == 'inception':
        model = torchvision.models.__dict__[args['model']](weights=Inception_V3_Weights.IMAGENET1K_V1)
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
        model = nn.DataParallel(model, args['gpus'])

        best_model_file = args['model_path']
        checkpoint = torch.load(best_model_file, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        model.eval()

    elif args['base'] == 'vit':
        model = torchvision.models.__dict__[args['model']](weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_ftrs = model.heads.head.in_features
        model.heads = nn.Linear(num_ftrs, 1)
        model.to(devices[0])
        model = nn.DataParallel(model, device_ids=devices)

        best_model_file = args['model_path']
        checkpoint = torch.load(best_model_file, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        model.eval()

    return model


def plot(x_list, y_list, x_label, y_label, title, plot_path):
    plt.plot(x_list, y_list)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(plot_path)
    plt.close()