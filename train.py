import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from config import *
from dataset import Lunch500
from model import *

def train(model, device, train_loader, optimizer, criterion):
    model.train()

    loss_total = 0

    for inputs, labels in train_loader:
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

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        loss = criterion(output, labels)
        loss_total += loss.item()

    loss_avg = loss_total / len(val_loader)

    return loss_avg

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using: {device}')

    # dataset
    dataset_dir = os.path.join('lunch500')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    
    lunch500_train = Lunch500(root_dir=dataset_dir, mode='train',transform=transform)
    lunch500_val = Lunch500(root_dir=dataset_dir, mode='valid',transform=transform)

    # data loader
    loader_train = DataLoader(lunch500_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    loader_val = DataLoader(lunch500_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # load model
    out_features = len(label_names)
    model = ResNet18(out_features=out_features, pretrained=True)
    mdoel = model.to(device)

    # train
    save_path = f'{model.name}_results'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print('make path')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    loss_train_list = []
    loss_val_list = []
    loss_val_best = float('inf')
    for epoch in tqdm(range(epoch_num), leave=False):
        loss_train = train(model, device, loader_train, optimizer, criterion)
        loss_val = valid(model, loader_val, criterion, device)

        loss_train_list.append(loss_train)
        loss_val_list.append(loss_val)

        if loss_val < loss_val_best:
            torch.save(model.state_dict(), f'./{save_path}/{model.name}_best.pt')
            loss_val_best = loss_val

    model.save_model(f'./{save_path}/{model.name}_last.pt')

    fig = plt.figure()
    plt.title('Training loss')
    plt.plot(loss_val_list, 'r', label='valid')
    plt.plot(loss_train_list, 'b', label='train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    fig.savefig('model_loss.png')