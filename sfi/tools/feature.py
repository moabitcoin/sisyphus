import tqdm
from torch.utils.data import DataLoader

from einops import reduce
from pathlib import Path

from sfi.datasets import ImageDirectory
from sfi.features import FeatureExtractor
from sfi.io import ArrayIO


def main(args):

    # Support features extraction without image resizing
    image_size = None if args.batch == 1 else args.image_size
    extract = FeatureExtractor(image_size=image_size)

    dataset = ImageDirectory(root=args.images, transform=extract.transform)
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=16)

    for images, paths in tqdm.tqdm(loader, desc='images', unit="batch", ascii=True):

        paths = map(Path, paths)

        n, c, h, w = images.size(0), 2048, images.size(2), images.size(3)

        # resnet5 downsamples x2 five times
        h, w = h // 32, w // 32

        # MAC feature descriptor
        features = extract(images)
        features = reduce(features, "n (h w) c -> n c", "max", n=n, h=h, w=w, c=c)
        features = features.data.cpu().numpy()

        filenames = [args.features.joinpath(p.stem + '.npy') for p in paths]

        _ = [ArrayIO.save(a, b) for a, b in zip(filenames, features)]
