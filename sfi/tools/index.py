import sys
from pathlib import Path

from torch.utils.data import DataLoader, random_split

import numpy as np
from einops import reduce

from faiss import IndexFlatL2, IndexIVFPQ

from tqdm import tqdm

from sfi.datasets import FeatureDirectory
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

    dataset = FeatureDirectory(root=args.features)
    train_dataset, index_dataset = random_split(dataset, [args.num_train, len(dataset) - args.num_train])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    index_loader = DataLoader(index_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    N, C = len(train_dataset), args.dim

    train_features = np.empty(shape=(N, C), dtype=np.float32)

    for i, (features, paths) in enumerate(tqdm(train_loader, desc="Train", unit="batch", ascii=True)):

        train_features[i * args.batch_size: i * args.batch_size + args.batch_size] = features

    quantizer = IndexFlatL2(C)

    index = IndexIVFPQ(quantizer, C, kNumCells, kNumCentroids, kNumBitsPerIdx)
    index.do_polysemous_training = True

    print("Training index on training features", file=sys.stderr)
    index.train(train_features)

    metadata = []

    for features, paths in tqdm(index_loader, desc="Index", unit="batch", ascii=True):

        features = np.ascontiguousarray(features)
        index.add(features)

        for path in paths:
            metadata.append(path)

    print('Saving index file and metadata')
    IndexIO.save(args.index.with_suffix(".idx"), index)
    JsonIO.save(args.index.with_suffix(".json"), metadata)
