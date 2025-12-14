import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='Noto Sans TC'  # for chinese

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
import numpy as np
from tqdm import tqdm
import random
import argparse

from config import *
from dataset import *
from model import *

@torch.no_grad()
def test(model, val_loader, criterion, device):
    model.eval()

    all_labels = []
    all_preds = []

    loss_total = 0
    correct = 0
    correct_top_5 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            output = model(inputs)
            loss = criterion(output, labels)
            loss_total += loss.item()

            pred = output.argmax(dim=1)
            pred_top_5 = output.topk(5).indices

            total += labels.size(0)
            correct += (pred == labels).sum().item()
            correct_top_5 += pred_top_5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    loss_avg = loss_total / len(val_loader)
    accuracy = 100 * correct / total
    accuracy_top_5 = 100 * correct_top_5 / total
    
    return loss_avg, accuracy, accuracy_top_5, cm

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
        transforms.Resize((224, 224)),
    ])
    
    lunch500_val = Lunch500(root_dir=dataset_dir, mode='valid',transform=transform)

    # data loader
    loader_val = DataLoader(lunch500_val, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    # model
    label_names = lunch500_val.labels
    out_features = len(label_names)
    Model_builder = get_Model(choose_model)
    model = Model_builder(out_features=out_features, pretrained=True).to(device)

    if use_flip:
        model.name += '_HorizontalFlip'
    if use_cutmix:
        model.name += '_CutMix'
        
    model.load_state_dict(torch.load(f'./{model.name}_results/{model.name}_best.pt', weights_only=True))
    print(f'testing: {model.name}')

    # test on the validation set
    criterion = torch.nn.CrossEntropyLoss()
    loss_val, accuracy, accuracy_top_5, cm = test(model, loader_val, criterion, device)
    print(f'loss_val: {loss_val}, accuracy: {accuracy:.2f}%, top-5 accuracy: {accuracy_top_5:.2f}%')

    # testing on random images
    with torch.no_grad():
        test_idx = random.randint(0, len(lunch500_val))
        img, label = lunch500_val[test_idx]
        img = img.to(device)
        img = img.unsqueeze(0)

        # inference
        pred = model(img)[0]
        pred = softmax(pred, dim=0).cpu().numpy()

        # visualize
        # img = img[0].permute((1, 2, 0)).cpu().numpy()
        # img = (img * 255).astype(np.uint8)

        # fig = plt.figure(figsize=(32, 9))
        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.title(f'label: {label_names[label]}')

        # plt.subplot(1, 2, 2)
        # plt.bar(label_names, pred)
        # plt.title('prediction')
        # plt.tight_layout()
        # fig.savefig(f'test_{model.name}.png')
        
        # confusion matrix
        plt.figure(figsize=(15, 12))
        ConfusionMatrixDisplay(cm).plot()
        plt.xticks(ticks=range(0, len(label_names)), labels=label_names, rotation='vertical')
        plt.yticks(ticks=range(0, len(label_names)), labels=label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix of {model.name}')
        plt.tight_layout()
        plt.savefig(f'./{model.name}_results/confusion_matrix_{model.name}.png')