import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader

from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop

from tqdm import tqdm

from sfi.io import JsonIO
from sfi.datasets import ImageDirectory
from sfi.transforms import ToImageMode


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
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=mean, std=std)])

    dataset = ImageDirectory(root=args.dataset, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Binary classifier on top of resnet50
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)
    model = nn.DataParallel(model)

    # Restore trained weights
    weights = torch.load(str(args.model), map_location=device)
    model.load_state_dict(weights)

    model.eval()

    results = []

    with torch.no_grad():
        for inputs, paths in tqdm(dataloader, desc="infer", unit="batch", ascii=True):
            inputs = inputs.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, dim=1)
            preds = preds.data.cpu().numpy()

            probs = nn.functional.softmax(outputs, dim=1)
            probs = probs.data.cpu().numpy()

            for path, pred, prob in zip(paths, preds, probs):
                result = {"class": pred.item(), "probability": round(prob.max().item(), 3), "path": Path(path).name}
                results.append(result)

    JsonIO.save(args.results, results)
