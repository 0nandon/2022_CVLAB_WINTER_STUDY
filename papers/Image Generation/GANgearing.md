
# GAN-Supervised Dense Visual Alignment

[paper link here](https://arxiv.org/pdf/2112.05143.pdf)

## Abstract

We propose GAN-Supervised Learning, a framework for learning discriminative models and their GAN-generated training
data jointly end-to-end.

Inspired by the classic Congealing method, our GANgealing algorithm trains a Spatial Transformer
to map `random samples from a GAN trained on unaligned data to a common.`

## Introduction

In this paper, we take inspiration from a series of classic works on automatic joint image set alignment.

While congealing can work surprisingly well on simple binary images, such as MNIST digits, the direct pixel-level alignment
is not powerful enough to handle most datasets with `significant apperance` and `pose variation.`

To address these limitations, we propose GANgealing:
a GAN-supervised algorithm that learns transformations of input images to bring them into better join alignment.


