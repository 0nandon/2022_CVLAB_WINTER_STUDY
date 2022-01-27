
# Spatial Transformer Networks

[paper link here](https://arxiv.org/pdf/1506.02025.pdf)

[tutorial code here](https://tutorials.pytorch.kr/intermediate/spatial_transformer_tutorial.html)

## Abstract

CNN define an exceptionally powerful class of models but are still limited by the lack of ability to be
`spatially invariant` to the input data in a computationally and parameter efficient manner.

In this work, we introduce a new learnable module, the `Spatial Transformer`, which `explicitly allows the
spatial manipulation of data` within the network.

THis differentiable module can be inserted into existing convolutional architectures, giving neural
networks the ability to actively transform feature maps, conditional on the feature map itself,
`without any extra training supervision or modification to the optimisation process.`

## Introduction
