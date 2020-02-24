#!/usr/bin/env python3

import sys

from PIL import Image
from PIL import ImageDraw


if len(sys.argv) != 3:
    sys.exit("Usage: {} in.jpg out.jpg".format(sys.argv[0]))


infile = sys.argv[1]
outfile = sys.argv[2]

image = Image.open(infile)
w, h = image.size

draw = ImageDraw.Draw(image)

n = 7

dx = w // n
dy = h // n

for i in range(1, n):
    draw.line([i * dx, 0, i * dx, h], fill="green", width=1)

for j in range(1, n):
    draw.line([0, j * dy, w, j * dy], fill="green", width=1)

image.save(outfile, optimize=True)
