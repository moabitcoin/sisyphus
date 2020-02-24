import sys
from pathlib import Path

from torch.utils.data import DataLoader, random_split

import numpy as np
from einops import reduce

from faiss import IndexFlatL2, IndexIVFPQ

from tqdm import tqdm

from sfi.datasets import ImageDirectory
from sfi.features import FeatureExtractor
from sfi.io import IndexIO, JsonIO

kNumCells = 100
kNumCentroids = 256  # Note: on gpu this will not work; see links below
kNumBitsPerIdx = 8

# Gpu centroid limitations
# - https://github.com/facebookresearch/faiss/blob/a8118acbc516b0263dde610862c806400cc48bf5/gpu/impl/IVFPQ.cu#L69-L92
# - https://github.com/facebookresearch/faiss/blob/a8118acbc516b0263dde610862c806400cc48bf5/ProductQuantizer.cpp#L189


def main(args):
    # https://github.com/facebookresearch/faiss/blob/a8118acbc516b0263dde610862c806400cc48bf5/Clustering.cpp#L78-L80
    if args.num_train < max(kNumCells, kNumCentroids):
        sys.exit("Error: require at least {} training samples".format(max(kNumCells, kNumCentroids)))

    extract = FeatureExtractor(image_size=args.image_size)

    dataset = ImageDirectory(root=args.frames, transform=extract.transform)
    train_dataset, index_dataset = random_split(dataset, [args.num_train, len(dataset) - args.num_train])

    if len(train_dataset) > len(index_dataset) or len(train_dataset) > 0.25 * len(index_dataset):
        sys.exit("Error: training dataset too big: train={}, index={}".format(len(train_dataset), len(index_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    index_loader = DataLoader(index_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    N, C = len(train_dataset), 2048

    train_features = np.empty(shape=(N, C), dtype=np.float32)

    for i, (images, paths) in enumerate(tqdm(train_loader, desc="Train", unit="batch", ascii=True)):
        n, h, w = images.size(0), images.size(2), images.size(3)

        features = extract(images)
        features = features.data.cpu().numpy()

        # resnet5 downsamples x2 five times
        h, w = h // 32, w // 32

        # MAC feature
        features = reduce(features, "n (h w) c -> n c", "max", n=n, h=h, w=w, c=C)

        train_features[i * args.batch_size: i * args.batch_size + n] = features

    quantizer = IndexFlatL2(C)

    index = IndexIVFPQ(quantizer, C, kNumCells, kNumCentroids, kNumBitsPerIdx)
    index.do_polysemous_training = True

    print("Training index on training features", file=sys.stderr)
    index.train(train_features)

    metadata = []

    for images, paths in tqdm(index_loader, desc="Index", unit="batch", ascii=True):
        n, h, w = images.size(0), images.size(2), images.size(3)

        # resnet5 downsamples x2 five times
        h, w = h // 32, w // 32

        # MAC feature descriptor
        features = extract(images)
        features = reduce(features, "n (h w) c -> n c", "max", n=n, h=h, w=w, c=C)
        features = features.data.cpu().numpy()

        # C-array required for faiss FFI: tensors might not be contiguous
        features = np.ascontiguousarray(features)

        # Add a batch of (batch*49, 2048) unpooled features to the index at once
        index.add(features)

        for path in paths:
            fname = Path(path).name
            metadata.append(fname)

    IndexIO.save(args.index.with_suffix(".idx"), index)
    JsonIO.save(args.index.with_suffix(".json"), metadata)
