
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

The Key is in `employing the latent space of a GAN` to automatically generate paired training data for a Spatial Transformer.

## GAN-Supervised Learning

In this section, we present GAN-Supervised Learning.

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/GANgearing_2.png" width=600>

Under this framework, (*x, y*) pairs are sampled from a pretrained GAN generator, where *x* is a `random sample
from the GAN` and *y* is the `sample obtained by applying a learned latent manipulation` to *x*'s latent code.

This framework minimizes the following loss:

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/%20GANgearing_1.png" width=500>

### 3.1 Dense Visual Alignment


