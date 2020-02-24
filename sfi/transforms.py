import torchvision.transforms.functional as F


def to_image_mode(image, mode):
    return image.convert(mode)


class ToImageMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return to_image_mode(image, self.mode)


def pad_to_multiple(image, multiple, fill=0, padding_mode="constant"):
    w, h = image.size

    def next_multiple_of(n, multiple):
        return ((n // multiple) + int(bool(n % multiple))) * multiple

    padded_w = next_multiple_of(w, multiple)
    padded_h = next_multiple_of(h, multiple)

    pad_left = (padded_w - w) // 2
    pad_right = pad_left + (padded_w - w) % 2

    pad_top = (padded_h - h) // 2
    pad_bottom = pad_top + (padded_h - h) % 2

    padding = (pad_left, pad_top, pad_right, pad_bottom)

    return F.pad(image, padding, fill=fill, padding_mode=padding_mode)


class PadToMultiple:
    def __init__(self, multiple, fill=0, padding_mode="constant"):
        self.multiple = multiple
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image):
        return pad_to_multiple(image, multiple=self.multiple, fill=self.fill,
                               padding_mode=self.padding_mode)
