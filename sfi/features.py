import sys

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

from einops import rearrange

from sfi.transforms import ToImageMode, PadToMultiple


class FeatureExtractor:
    def __init__(self, image_size=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            print("Using CUDA, benchmarking implementations", file=sys.stderr)
            torch.backends.cudnn.benchmark = True

        # Set up pre-trained resnet in inference mode
        resnet = resnet50(pretrained=True, progress=False)

        # Chop off classification head
        resnet.fc = nn.Identity()

        # In addition do not pool, keep spatial information if user wants to
        resnet.avgpool = nn.Identity()

        for params in resnet.parameters():
            params.requires_grad = False

        resnet = resnet.to(device)
        resnet = nn.DataParallel(resnet)

        resnet.eval()

        self.net = resnet
        self.device = device
        self.image_size = image_size

    @property
    def transform(self):
        # ImageNet statistics (because we use pre-trained model)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        transforms = [ToImageMode("RGB")]
        if self.image_size:
          transforms.append(Resize(self.image_size))

        # We zero pad (PadToMultiple) since resnet5 downsamples x2 five times

        transforms += [PadToMultiple(32, fill=0),
                       ToTensor(), Normalize(mean=mean, std=std)]

        return Compose(transforms)

    # batch of NCHW image tensors to batch of NHWC feature tensors
    def __call__(self, images):
        n, c, h, w = images.size(0), 2048, images.size(2), images.size(3)

        assert h % 32 == 0, "height divisible by 32 for resnet50"
        assert w % 32 == 0, "width divisible by 32 for resnet50"

        with torch.no_grad():
            images = images.to(self.device)

            # resnet5 downsamples x2 five times
            h, w = h // 32, w // 32

            # resnet50 outputs flat view over a batch with 2048 channels, spatial resolution HxW
            # https://github.com/pytorch/vision/blob/ac2e995a4352267f65e7cc6d354bde683a4fb402/torchvision/models/resnet.py#L202-L204

            features = self.net(images)
            features = rearrange(features, "n (c h w) -> n (h w) c", n=n, h=h, w=w, c=c)

            return features
