from torch.utils.data import DataLoader

from einops import reduce

from sfi.datasets import ImageSingleton
from sfi.features import FeatureExtractor
from sfi.io import ArrayIO


def main(args):
    extract = FeatureExtractor(image_size=args.image_size)

    # We use this tool to compute query features on images of arbitrary sizes.
    # That's why we can not batch images and have to feed them one by one.

    dataset = ImageSingleton(root=args.frame, transform=extract.transform)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    for images, paths in loader:
        assert images.size(0) == 1, "image batch size of one for required"

        n, c, h, w = images.size(0), 2048, images.size(2), images.size(3)

        # resnet5 downsamples x2 five times
        h, w = h // 32, w // 32

        # MAC feature descriptor
        features = extract(images)
        features = reduce(features, "n (h w) c -> n c", "max", n=n, h=h, w=w, c=c)
        features = features.data.cpu().numpy()

        ArrayIO.save(args.feature, features[0])
