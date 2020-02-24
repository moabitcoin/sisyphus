import sys

from torch.utils.data import DataLoader

from PIL import Image

import numpy as np
from einops import reduce

from skvideo.io import vread

from sfi.features import FeatureExtractor
from sfi.utils import batched


def main(args):
    args.frames.mkdir(exist_ok=True)

    key = None
    video = vread(str(args.video))
    extract = FeatureExtractor(image_size=args.image_size)

    nframes, nkeys = 0, 0

    for i, batch in enumerate(batched(video, args.batch_size)):
        # We should use the IterableDataset from upcoming PyTorch version for FramesDataset

        frames = [Image.fromarray(each) for each in batch]

        dataset = [extract.transform(frame) for frame in frames]
        dataloader = DataLoader(dataset, batch_size=args.batch_size)

        assert len(dataloader) == 1
        images = next(iter(dataloader))

        n, c, h, w = images.size(0), 2048, images.size(2), images.size(3)

        features = extract(images)
        features = features.data.cpu().numpy()

        # resnet5 downsamples x2 five times
        h, w = h // 32, w // 32

        features = reduce(features, "n (h w) c -> n c", reduction=args.pool, n=n, h=h, w=w, c=c)

        for j, (frame, feature) in enumerate(zip(frames, features)):
            nframes += 1

            fid = i * args.batch_size + j

            if key:
                prev_frame, prev_feature = key

                if similarity(prev_feature, feature) > args.similarity:
                    continue

            nkeys += 1
            key = frame, feature
            frame.save(args.frames / "{:010d}.jpg".format(fid))

    if nframes != 0:
        print("Processed total={} keep={} drop={} ratio={}"
              .format(nframes, nkeys, nframes - nkeys, round(nkeys / nframes, 2)), file=sys.stderr)


def similarity(x, y):
    return (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
