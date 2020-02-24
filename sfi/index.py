import numpy as np
from einops import rearrange

from faiss import IndexPQ

from sfi.io import IndexIO, JsonIO


# TODO: benchmark
kNumResultsPerIndex = 512


class IndexQueryError(Exception):
    pass


class Index:
    def __init__(self, path, metadata, features_size, num_probes=1):
        self.index = IndexIO.load(path)
        self.index.nprobes = num_probes

        # Disable Polysemous Codes until we know threshold for MACs
        # self.index.search_type = IndexPQ.ST_polysemous
        # self.index.polysemous_ht = 768

        self.metadata = JsonIO.load(metadata)
        self.features_size = features_size

    def query(self, query, num_results=1):
        N, C = query.shape

        if N != self.features_size * self.features_size:
            raise IndexQueryError("query feature size does not match index feature size")

        # C-array required for faiss FFI: tensors might not be contiguous
        query = np.ascontiguousarray(query)

        dists, idxs = self.index.search(query, kNumResultsPerIndex)

        dists = rearrange(dists, "() n -> n")
        idxs = rearrange(idxs, "() n -> n")

        results = list(zip(dists, idxs))

        _, uniqued = np.unique([i for _, i in results], return_index=True)
        results = [results[i] for i in uniqued]
        results = sorted(results, key=lambda v: v[0])

        results = [(round(d.item(), 3), self.metadata[i])
                   for d, i in results[:num_results]]

        return results
