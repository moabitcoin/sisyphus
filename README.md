<h1 align='center'>:mount_fuji: Sisyphus</h1>

Scene similarity for weak object discovery & classification. Labelling images for building an object classification models is a labor intensive task (a.k.a [rolling a ball uphill](https://en.wikipedia.org/wiki/Sisyphus)). This repository leverages [image similarity](https://www.github.com/facebookresearch/faiss) to generate a weakly labeled dataset which can be cleaned up order of magnitude faster than going through the whole image/video corpus. The expected speedup is inversely proportional to rarity of the object of interest. We also provide a [blockages/construction](https://github.com/moabitcoin/sisyphus/releases/tag/v1.0.0) detection model trained on drive data from [Berlin.](https://hoodmaps.com/berlin-neighborhood-map) and tools contained in this repository.

# Table of Contents
* [Installation](#computer-installation)
* [Sample usage](#tada-usage)
  - [Features Extraction](#architectures)
  - [Building faiss index](#environments)
  - [Loading faiss index](#environments)
  - [Querying faiss index](#environments)
* [Video to keyframes]()
* [Case study]()
  - [Road blockages]()

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
    ./bin/sfi <tool> <args>
    ./bin/sfi --help
    ./bin/sfi <tool> --help
```

### Feature extraction

Extracts high level feature maps for all image frames from a trained convolutional neural net.
Saves `.npy` files with the extracted feature maps in parallel to all image frames.
We recommend running this step on GPUs.


### Building index

Builds an index from the `.npy` feature maps for fast and efficient approximate nearest neighbour queries based on L2 distance. The `quantizer` for the index needs to get trained on a small subset of the feature maps to approximate the dataset's centroids. Depending on the feature map's spatial resolution (pooled vs. unpooled) we build and save multiple indices (one per `depthwise` feature map axis).

### Load index

Loads up the index (slow) and keeps it in memory to handle nearest neighbour queries (fast).
Responds to queries by searching the index, aggregating results, and re-ranking them.

### Query index

Sends nearest neighbour requests against the query server and reports results to the user.
The query and results are based on the `.npy` feature maps.

### Frames vs. videos

The semantic frame index can work with image frames; for videos you should extract key frames first

```
    ./scripts/video-to-key-frames /path/to/video /tmp/frames/
```
The semantic frame index query can return key frame images; for inspection and sharing you should create a video
```
    ./scripts/key-frames-to-video /tmp/result/ nearest.mp4
```
For indexing and querying video sequences directly see our companion project for [video summarization](https://github.com/moabitcoin/Adversarial-video-summarization-pytorch).


## Case study
### Blockage detection

We record tens of thousand hours of drive video data and need to be able to search for semantically similar scenarios. Similarity could mean similar lighting conditions, similar vehicle types, similar traffic volumes, similar objects on the road, and so on.

### Implementation Sketch

In our first iteration we
- re-sample videos and extract their key frames
- for all key frames extract a trained convolutional neural net's high level feature maps
- index the feature maps for approximate nearest neighbor searches based on L2 distance
- query the indexed dataset for semantically similar scenarios

In our second iteration we work with videos directly; we
- re-sample videos and extract short sequences
- for all video sequences we compute a vector representation
- index the video sequence vector representations for approximate nearest neighbor searches based on L2 distance
- query the indexed dataset for semantically similar scenarios

See our companion project for [video summarization]https://github.com/moabitcoin/Adversarial-video-summarization-pytorch). Following the work in [arxiv.org/abs/1502.04681](https://arxiv.org/abs/1502.04681) we train a sequence model based auto-encoder for unsupervised video sequence vectors for indexing.

# References

- Product Quantizer (PQ) [part 1](http://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/), and [part 2](http://mccormickml.com/2017/10/22/product-quantizer-tutorial-part-2/)
- [Product Quantization for Nearest Neighbor Search](https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf)
- [Billion-scale similarity search with GPUs](https://arxiv.org/pdf/1702.08734.pdf)
- [faiss wiki](https://github.com/facebookresearch/faiss/wiki)
- [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681)
