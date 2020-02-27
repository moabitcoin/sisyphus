## :green_book: Case study
### :construction: Blockage detection

In this project we aimed at building a construction/blockages classier with minimal possible labelling efforts. We decided against manually sifting through 1000s of hours of video footage to source relevant training samples. Instead we used fast nearest neighbour search (FAISS) on image feature extracted from our video corpus and querying the corpus with known construction samples queries sourced from [Mapillary](https://www.mapillary.com/app/?lat=52.50721804777777&lng=13.329988131111122&z=17&pKey=VZ644ukdHzAbL4aEsyCBxw). This helped us build a quasi-clean training set which could be cleaned up minimal effort. After cleaning the training set we [train the linear layer]() of ResNet50 (pre-trained on ImageNet) to build a blockages/construction classier.

### :musical_score: Implementation Sketch

In our first iteration we
- [Re-sample the videos](https://github.com/moabitcoin/sisyphus/blob/master/scripts/video-to-key-frames) and extract key frames
- [Extract](https://github.com/moabitcoin/sisyphus#rocket-feature-extraction) [MAC features](https://arxiv.org/pdf/1511.05879.pdf) for all key frames with a pre-trained ResNet50 model
- Use the MAC features to [train/build](https://github.com/moabitcoin/sisyphus#vhs-load-index) an index with [FAISS](https://www.github.com/facebookresearch/faiss)
- [Query the index](https://github.com/moabitcoin/sisyphus#crystal_ball-query-index) for semantically similar scenarios sorted by approximate L2 distance

In our second iteration we work with videos directly; we
- Re-sample videos and extract short sequences ()~32 frames)
- Compute representations using a [pre-trained video model](https://github.com/moabitcoin/ig65m-pytorch)
- [Index the representations](https://github.com/moabitcoin/sisyphus#european_post_office-building-index) for approximate nearest neighbour search
- [Query the index](https://github.com/moabitcoin/sisyphus#crystal_ball-query-index) for semantically similar video sequences

See our companion project for [video summarisation](https://github.com/moabitcoin/Adversarial-video-summarization-pytorch). Following the work in [arxiv.org/abs/1502.04681](https://arxiv.org/abs/1502.04681) we train a sequence model based auto-encoder for unsupervised video sequence vectors for indexing & search.

### Querying index
Since we didn't have examples of blockages  / construction sites for Berlin and elsewhere. We sourced few construction samples from [mapillary](https://www.mapillary.com/app/?lat=52.50715057361111&lng=13.330102460277772&z=17&pKey=RT1cFReHJwMS8RqWz7_qFQ) and used them to query our index. Sample retrieved results are below. We use [query expansion](https://en.wikipedia.org/wiki/Query_expansion) to further improve our retrieval results. Below are some of the retrieved results from our corpus.

Retrieved results

<table><tr><td>
  <img src="https://github.com/moabitcoin/sisyphus/blob/master/assets/000000.jpg" width="960">
</td></tr></table>
<table><tr><td>
  <img src="https://github.com/moabitcoin/sisyphus/blob/master/assets/000001.jpg" width="960">
</td></tr></table>

### Classifier results
Blockage detection results after training the last linear layer resnet50 model.

<table><tr><td>
  <img src="https://github.com/moabitcoin/sisyphus/blob/master/assets/results-on-map.gif" width="960">
</td></tr></table>

### :bookmark: References

- Product Quantiser (PQ) [part 1](http://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/), and [part 2](http://mccormickml.com/2017/10/22/product-quantizer-tutorial-part-2/)
- [Product Quantisation for Nearest Neighbour Search](https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf)
- [Billion-scale similarity search with GPUs](https://arxiv.org/pdf/1702.08734.pdf)
- [Faiss wiki](https://github.com/facebookresearch/faiss/wiki)
- [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681)
