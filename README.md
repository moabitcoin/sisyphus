# Semantic Frame Index

Fast and efficient queries on video frames by semantic similarity.


## Use Case

We record tens of thousand hours of drive video data and need to be able to search for semantically similar scenarios.
Simlarity could mean similar lighting conditions, similar vehicle types, similar traffic volumes, similar objects on the road, and so on.


## Implementation Sketch

We
- extract key frames using a neural net for frame similarity in feature space
- extract a trained convolutional neural net's high level feature maps for all key frames
- compute Maximum Activations of Convolution (MAC) features from the high-level feature maps
- index the feature maps for approximate nearest neighbor searches based on L2 distance
- query the indexed dataset for semantically similar scenarios


## Usage

All tools can be invoked via

    ./bin/sfi <tool> <args>

    ./bin/sfi --help
    ./bin/sfi <tool> --help


### stream-index

Builds an index from a directory of images for fast and efficient approximate nearest neighbor queries based on L2 distance.
The quantizer for the index needs to get trained on a small subset of the feature maps to approximate the dataset's centroids.
We recommend runing this step on GPUs.


### save-feature

Extracts high level feature maps and computes MACs for an image frames from a trained convolutional neural net.


### save-frames

Extracts semantic key frames from videos based on a trained convolution net for feature similarity between frames.


### query-server

Loads up the index (slow) and keeps it in memory to handle nearest neighbor queries (fast).
Responds to queries by searching the index, aggregating results, and re-ranking them.


### query-client

Sends nearest neighbor requests against the query server and reports results to the user.
The query and results are based on the saved MAC features.


### model-train

Trains a binary classification model on a dataset (potentially noisy and obtained from the index).
We recommend runing this step on GPUs.


### model-infer

Predicts binary classification labels on a dataset, using a trained model.


## Development

Create a self-contained reproducible development environment

    make i

Get into the development environment

    make r

The Python source code directory is mounted into the container: if you modify it on the host it will get modified in the container.

To make data visible in the container set the datadir env var, e.g. to make your `/tmp` directory show up in `/data` inside the container run

    make r datadir=/tmp

See the `Makefile` for options and more advanced targets.


## References

- [Particular object retrieval with integral max-pooling of CNN activations](https://arxiv.org/abs/1511.05879)
- Product Quantizer (PQ) [part 1](http://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/), and [part 2](http://mccormickml.com/2017/10/22/product-quantizer-tutorial-part-2/)
- [Product Quantization for Nearest Neighbor Search](https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf)
- [Billion-scale similarity search with GPUs](https://arxiv.org/pdf/1702.08734.pdf)
- [faiss wiki](https://github.com/facebookresearch/faiss/wiki)


## License

Copyright © 2019 MoabitCoin

Distributed under the MIT License (MIT).
