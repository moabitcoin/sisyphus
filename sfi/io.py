import json

import numpy as np
import faiss


class ArrayIO:
    @staticmethod
    def save(path, x):
        return np.save(str(path), x, allow_pickle=False)

    @staticmethod
    def load(path):
        return np.load(str(path), allow_pickle=False)


class IndexIO:
    @staticmethod
    def save(path, x):
        return faiss.write_index(x, str(path))

    @staticmethod
    def load(path):
        return faiss.read_index(str(path))


class JsonIO:
    @staticmethod
    def save(path, x):
        with path.open("w") as fd:
            return json.dump(x, fd)

    @staticmethod
    def load(path):
        with path.open("r") as fd:
            return json.load(fd)
