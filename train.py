import torch
from torch.utils.data import DataLoader, default_collate
from torchvision import transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

from config import *
from dataset import Lunch500
from model import *

def train(model, device, train_loader, optimizer, criterion):
    model.train()

    loss_total = 0

    for inputs, labels in tqdm(train_loader, position=1, leave=False):
    # for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        loss = criterion(output, labels)
        loss_total += loss.item()
        
        loss.backward()
        optimizer.step()

    loss_avg = loss_total / len(train_loader)

    return loss_avg

@torch.no_grad()
def valid(model, val_loader, criterion, device):
    model.eval()

    loss_total = 0

    for inputs, labels in tqdm(val_loader, position=1, leave=False):
    # for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        loss = criterion(output, labels)
        loss_total += loss.item()

    loss_avg = loss_total / len(val_loader)

    return loss_avg

def get_Model(model):
    if model == 'ResNet18':
        return ResNet18
    elif model == 'MobileNet_V3_Small':
        return MobileNet_V3_Small

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ResNet18', 'MobileNet_V3_Small'], required=True)
    parser.add_argument('--flip', action='store_true', default=False)
    parser.add_argument('--cutmix', action='store_true', default=False)

    args = parser.parse_args()
    choose_model = args.model
    use_flip = args.flip
    use_cutmix = args.cutmix
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    # dataset
    dataset_dir = os.path.join('lunch500')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
    ]) if use_flip else transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    lunch500_train = Lunch500(root_dir=dataset_dir, mode='train',transform=transform)
    lunch500_val = Lunch500(root_dir=dataset_dir, mode='valid',transform=transform)

    # load model
    out_features = len(lunch500_train.labels)
    Model_builder = get_Model(choose_model)
    model = Model_builder(out_features=out_features, pretrained=True, freeze_pretrained=False).to(device)
    if use_flip:
        model.name += '_HorizontalFlip'
    if use_cutmix:
        model.name += '_CutMix'

    cutmix = v2.CutMix(num_classes=out_features)
    mixup = v2.MixUp(num_classes=out_features)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    # data loader
    loader_train = DataLoader(lunch500_train, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, shuffle=True, collate_fn=collate_fn if use_cutmix else None)
    loader_val = DataLoader(lunch500_val, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    # train
    save_path = f'{model.name}_results'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f'make path: {save_path}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    loss_train_list = []
    loss_val_list = []
    loss_val_best = float('inf')
    for epoch in tqdm(range(epoch_num), position=0, leave=True):
        loss_train = train(model, device, loader_train, optimizer, criterion)
        loss_val = valid(model, loader_val, criterion, device)

        loss_train_list.append(loss_train)
        loss_val_list.append(loss_val)

        if loss_val < loss_val_best:
            torch.save(model.state_dict(), f'./{save_path}/{model.name}_best.pt')
            loss_val_best = loss_val

    torch.save(model.state_dict(), f'./{save_path}/{model.name}_last.pt')
    
    fig = plt.figure()
    plt.title('Training loss')
    plt.plot(loss_val_list, 'r', label='valid')
    plt.plot(loss_train_list, 'b', label='train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    fig.savefig(f'./{save_path}/loss_{model.name}.png')