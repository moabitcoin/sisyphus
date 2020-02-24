import sys

import torch
import torch.nn as nn

from torchvision.models.video import r2plus1d_18

from einops import rearrange

from skvideo.io import vread

from sfi.utils import batched


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Using CUDA, benchmarking implementations", file=sys.stderr)
        torch.backends.cudnn.benchmark = True

    # r2d2 says "beep beep"
    resnet = r2plus1d_18(pretrained=True, progress=False)

    resnet.fc = nn.Identity()
    # resnet.avgpool = nn.Identity()

    for params in resnet.parameters():
        params.requires_grad = False

    resnet = resnet.to(device)
    resnet = nn.DataParallel(resnet)

    resnet.eval()

    # Pre-trained Kinetics-400 statistics for normalization
    mean, std = [0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]

    mean = rearrange(torch.as_tensor(mean), "n -> () n () ()")
    std = rearrange(torch.as_tensor(std), "n -> () n () ()")

    video = vread(str(args.video))

    with torch.no_grad():
        for i, batch in enumerate(batched(video, args.timesteps)):
            # TODO:
            # - encapsulate video dataset
            # - abstract away transforms
            # - fix timesteps vs batching

            batch = rearrange(batch, "t h w c -> t c h w")
            batch = torch.tensor(batch)
            batch = batch.to(torch.float32) / 255

            batch = (batch - mean) / std

            # model expects NxCxTxHxW
            inputs = rearrange(batch, "t c h w -> () c t h w")
            inputs = inputs.to(device)

            outputs = resnet(inputs)
            outputs = rearrange(outputs, "() n -> n")
            outputs = outputs.data.cpu().numpy()

            print("seq={}, frames=range({}, {}), prediction={}"
                  .format(i, i * args.timesteps, (i + 1) * args.timesteps, outputs.shape))
