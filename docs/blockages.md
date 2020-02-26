## :green_book: Case study
### :construction: Blockage detection

We record tens of thousand hours of drive video data and need to be able to search for semantically similar scenarios. Similarity could mean similar lighting conditions, similar vehicle types, similar traffic volumes, similar objects on the road, and so on.

### :musical_score: Implementation Sketch

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

See our companion project for [video summarization](https://github.com/moabitcoin/Adversarial-video-summarization-pytorch). Following the work in [arxiv.org/abs/1502.04681](https://arxiv.org/abs/1502.04681) we train a sequence model based auto-encoder for unsupervised video sequence vectors for indexing & search.

## :bookmark: References

- Product Quantizer (PQ) [part 1](http://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/), and [part 2](http://mccormickml.com/2017/10/22/product-quantizer-tutorial-part-2/)
- [Product Quantization for Nearest Neighbor Search](https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf)
- [Billion-scale similarity search with GPUs](https://arxiv.org/pdf/1702.08734.pdf)
- [faiss wiki](https://github.com/facebookresearch/faiss/wiki)
- [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681)
