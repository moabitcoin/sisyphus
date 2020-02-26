from PIL import Image

from torch.utils.data import Dataset

from sfi.utils import files
from sfi.io import ArrayIO

# PyTorch can not transport a Path object through data loaders.
# Serialize Path to str here; users have to encode via Path(path).


class ImageDirectory(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.paths = files(root)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = str(self.paths[i])
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, path


class ImageSingleton(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.path = root
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, i):
        assert i == 0

        path = str(self.path)
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, path


class FeatureDirectory(Dataset):
    def __init__(self, root):
        super().__init__()

        self.paths = files(root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = str(self.paths[i])
        f = ArrayIO.load(path)

        return f, path
