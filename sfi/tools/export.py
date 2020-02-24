import torch
import torch.onnx
import torch.nn as nn

from torchvision.models import resnet50

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Using CUDA, benchmarking implementations", file=sys.stderr)
        torch.backends.cudnn.benchmark = True

    # Binary classifier on top of resnet50
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)
    model = nn.DataParallel(model)

    # Restore trained weights
    weights = torch.load(str(args.model), map_location=device)
    model.load_state_dict(weights)

    # Run dummy batch through model to trace computational graph
    batch = torch.rand(1, 3, 224, 224, device=device)

    torch.onnx.export(model.module, batch, str(args.onnx))
