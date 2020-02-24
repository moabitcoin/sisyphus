import sys
import copy
import collections

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn
from torch.utils.data import DataLoader

from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomHorizontalFlip

from tqdm import tqdm

from sfi.transforms import ToImageMode
from sfi.mixup import MixupDataLoaderAdaptor, MixupCrossEntropyLossAdaptor
from sfi.utils import decay_weights


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Using CUDA, benchmarking implementations", file=sys.stderr)
        torch.backends.cudnn.benchmark = True

    # ImageNet statistics (because we use pre-trained model)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = Compose([
        ToImageMode("RGB"),
        Resize(256),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std)])

    train_dataset = ImageFolder(root=args.dataset / "train", transform=transform)
    val_dataset = ImageFolder(root=args.dataset / "val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    train_loader = MixupDataLoaderAdaptor(train_loader)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = resnet50(pretrained=True, progress=False)

    # Add binary classification head
    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)
    model = nn.DataParallel(model)

    if args.resume_from:
        weights = torch.load(str(args.resume_from), map_location=device)
        model.load_state_dict(weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    counts = collections.Counter(train_dataset.targets).values()
    weight = torch.tensor([min(counts) / v for v in counts]).to(device)

    train_criterion = MixupCrossEntropyLossAdaptor(weight=weight)
    val_criterion = nn.CrossEntropyLoss(weight=weight)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch, args.num_epochs - 1))
        print("-" * 10)

        loss, _, _, _ = train(model, train_criterion, optimizer, device,
                              dataset=train_dataset, dataloader=train_loader)

        print("train loss: {:.4f}".format(loss))

        loss, acc, precision, recall = validate(model, val_criterion, device,
                                                dataset=val_dataset, dataloader=val_loader)

        print("val loss: {:.4f} acc: {:.4f} precision: {:.4f} recall: {:.4f}".format(loss, acc, precision, recall))

        if acc > best_acc:
            best_acc = acc
            best_wts = copy.deepcopy(model.state_dict())

        print()

    print("Best acc: {:4f}".format(best_acc))

    torch.save(best_wts, str(args.model))


def train(model, criterion, optimizer, device, dataset, dataloader):
    model.train()

    running_loss = 0.0

    for inputs, t, labels1, labels2 in tqdm(dataloader, desc="train", unit="batch", ascii=True):
        inputs = inputs.to(device)
        t = t.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, t, labels1, labels2)

        loss.backward()
        decay_weights(optimizer, 1e-4)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataset)

    return epoch_loss, float("NaN"), float("NaN"), float("NaN")


def validate(model, criterion, device, dataset, dataloader):
    model.eval()

    running_loss = 0.0
    tn, fn, tp, fp = 0, 0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="val", unit="batch", ascii=True):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            confusion = preds.float() / labels.float()
            tn += torch.sum(torch.isnan(confusion)).item()
            fn += torch.sum(confusion == float("inf")).item()
            tp += torch.sum(confusion == 1).item()
            fp += torch.sum(confusion == 0).item()

    epoch_loss = running_loss / len(dataset)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return epoch_loss, accuracy, precision, recall
