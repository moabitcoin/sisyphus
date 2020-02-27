<h1 align='center'>:mount_fuji: Sisyphus</h1>

Scene similarity for weak object discovery & classification. Labelling images for building an object classification models is a labor intensive task (a.k.a [rolling a ball uphill](https://en.wikipedia.org/wiki/Sisyphus)). This repository leverages [image similarity](https://www.github.com/facebookresearch/faiss) to generate a weakly labeled dataset which can be cleaned up order of magnitude faster than going through the whole image/video corpus. The expected speedup is inversely proportional to rarity of the object of interest. We also provide a [blockages/construction](https://github.com/moabitcoin/sisyphus/releases/tag/v1.0.0) detection model trained on drive data from [Berlin.](https://hoodmaps.com/berlin-neighborhood-map) and tools contained in this repository.

# Table of Contents
* [Installation](#computer-installation)
* [Sample usage](#tada-usage)
  - [Features Extraction](#rocket-feature-extraction)
  - [Building faiss index](#european_post_office-building-index)
  - [Loading faiss index](#vhs-load-index)
  - [Querying faiss index](#crystal_ball-query-index)
* [Video to keyframes](#camera-frames-vs-video_camera-videos)
* [Example: Traffic Blockages](./docs/blockages.md)

## :computer: Installation

Create a self-contained reproducible development environment & Get into the development environment

Example for running on CPUs:
```
make install dockerfile=Dockerfile.cpu dockerimage=moabitcoin/sfi-cpu
make run dockerimage=dockerimage=moabitcoin/sfi-cpu
```


Example for running on GPUs via [nvidia-docker](https://github.com/NVIDIA/nvidia-docker):
```
make install dockerfile=Dockerfile.gpu dockerimage=moabitcoin/sfi-gpu
make run dockerimage=dockerimage=moabitcoin/sfi-gpu runtime=nvidia
```

The Python source code directory is mounted into the container: if you modify it on the host it will get modified in the container, so you don't need to rebuild the image. To make data visible in the container set the datadir env var, e.g. to make your `/tmp` directory show up in `/data` inside the container run
```
make run datadir=/tmp
```
See the [`Makefile`](./Makefile) for options and more advanced targets.

## :tada: Usage

All tools can be invoked via
```
./bin/sfi --help
usage: sficmd [-h]  ...

optional arguments:
  -h, --help         show this help message and exit

commands:

    frames-extract   Extract video key frames w/intra frame similarity
    feature-extract  Extract image features w/ pre-trained resnet50
    feature-extract-vid
                     Extract features from videos wth 2(D+1) video model
    build-index      Builds a faiss index
    serve-index      Starts up the index http server
    query-index      Queries the index server for nearest neighbour
    model-train      Trains a classifier model with a resnet50 backbone
    model-infer      Runs inference with a classifier model
    model-export     Export a classifier model to onnx
```

### :rocket: Feature extraction

```
  ./bin/sfi feature-extract --help
```
Extracts high level [MAC](https://arxiv.org/pdf/1511.05879.pdf) feature maps for all image frames from a pre-trained convolutional neural net(ResNet-50 + ILSVRC2012). Save the features in individual `.npy` files with the extracted feature maps in parallel to all image frames. We recommend running this step on GPUs.

### :european_post_office: Building index
```
  ./bin/sfi index-build --help
```
Builds an index from the `.npy` feature maps for fast and efficient approximate nearest neighbour queries based on L2 distance. The `quantizer` for the index needs to get trained on a small subset of the feature maps to approximate the dataset's centroids. Depending on the feature map's spatial resolution (pooled vs. unpooled) we build and save multiple indices (one per `depthwise` feature map axis).

### :vhs: Load index
```
  ./bin/sfi index-serve --help
```
Loads up the index (slow) and keeps it in memory to handle nearest neighbour queries (fast).
Responds to queries by searching the index, aggregating results, and re-ranking them.

### :crystal_ball: Query index
```
  ./bin/sfi index-query --help
```
Sends nearest neighbour requests against the query server and reports results to the user.
The query and results are based on the `.npy` feature maps on which the index was build. The mapping from `.npy` files and images is saved in <index_file>.json.

### :camera: Frames vs. :video_camera: videos

Sisyphus works with images; for videos you should extract key frames first

```
  ./scripts/video-to-key-frames /path/to/video /tmp/frames/
```
The semantic frame index query can return key frame images; for inspection and sharing you should create a video
```
  ./scripts/key-frames-to-video /tmp/result/ nearest.mp4
```
For indexing and querying video sequences directly see our companion project for [unsupervised video summarisation](https://github.com/moabitcoin/Adversarial-video-summarization-pytorch). Another option would be to use `feature-extract-vid` (Disclaimer : Experimental)
