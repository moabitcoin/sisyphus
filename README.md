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
* [Case study: Traffic Blockages](./docs/blockages.md)

## :computer: Installation

Create a self-contained reproducible development environment

```
    make install
```
Get into the development environment
```
    make run
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
      -h, --help      show this help message and exit

    commands:

        save-frames   saves key frames for video
        stream-index  builds an index in streaming mode
        save-feature  saves features for frames
        save-feature3d
                      saves features for videos
        query-server  starts up the index query http server
        query-client  queries the query server for similar features
        model-train   trains a classifier model
        model-infer   runs inference with a classifier model
        model-export  export a classifier model to onnx
```

### :rocket: Feature extraction

```
  ./bin/sfi save-feature --help
```

Extracts high level feature maps for all image frames from a trained convolutional neural net. (ResNet-50)
Saves `.npy` files with the extracted feature maps in parallel to all image frames. We recommend running this step on GPUs.


### :european_post_office: Building index

Builds an index from the `.npy` feature maps for fast and efficient approximate nearest neighbour queries based on L2 distance. The `quantizer` for the index needs to get trained on a small subset of the feature maps to approximate the dataset's centroids. Depending on the feature map's spatial resolution (pooled vs. unpooled) we build and save multiple indices (one per `depthwise` feature map axis).

### :vhs: Load index

Loads up the index (slow) and keeps it in memory to handle nearest neighbour queries (fast).
Responds to queries by searching the index, aggregating results, and re-ranking them.

### :crystal_ball: Query index

Sends nearest neighbour requests against the query server and reports results to the user.
The query and results are based on the `.npy` feature maps.

### :camera: Frames vs. :video_camera: videos

The semantic frame index can work with image frames; for videos you should extract key frames first

```
    ./scripts/video-to-key-frames /path/to/video /tmp/frames/
```
The semantic frame index query can return key frame images; for inspection and sharing you should create a video
```
    ./scripts/key-frames-to-video /tmp/result/ nearest.mp4
```
For indexing and querying video sequences directly see our companion project for [video summarization](https://github.com/moabitcoin/Adversarial-video-summarization-pytorch).
