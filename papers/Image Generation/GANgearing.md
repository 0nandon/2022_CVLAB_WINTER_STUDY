
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

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/GANgearing_2.png" width=700>

Under this framework, (*x, y*) pairs are sampled from a pretrained GAN generator, where *x* is a `random sample
from the GAN` and *y* is the `sample obtained by applying a learned latent manipulation` to *x*'s latent code.

This framework minimizes the following loss:

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/%20GANgearing_1.png" width=500>

### 3.1 Dense Visual Alignment

GANgearing begins by training a latent variable generative model *G* on an unaligned input dataset.

With G trained, we are `free to draw samples` from the unaligned distribution by computing *x* = *G*(W)
for randomly sampled w ~ W, where W denotes the distribution over latents.

Now, consider a fixed latent vector c ∈ R<sup>512</sup>. This vector corresponds to a fixed synthetic image
*G*(c) from the original unaligned distribution.

We learn a Spatial Transformer *T* that is trained to wrap every random unaligned image *x* = *G*(w) to the
same target image *y* = *G*(c). Therefore, we can optimize the following loss:

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/GANgearing_3.png" width=300>

This simple approach is reasonable for datasets with `limited diversity`; however, `in the presence of significant
apperance and pose variation`, it is not reasonable to expect every unaligned sample.

Instead of using the same target *G*(c) for every randomly sampled image *G*(w), we mix c, w:
> mix(c, w) ∈ R<sup>512</sup>

to construct a per-sample target that retains appearance of *G*(w) but where the pose ane orientation of
the object in the target image is roughly identical across targets.

This give rise to the GANgearing loss function:

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/GANgearing_4.png" width=400>

#### Spatial Trnasformer Parameterization

We must choose how to constrain the *g* regressed by Spatial Transformer T.

Our final T is a composition of the `similarity Spatial Transformer` into the `unconstrained Spatial Transformer`,
which we found worked best.

When using the unconstrained *T*, it can be beneficial to add a total variation regularizer that encourages the predicted
flow to be smooth to mitigate degenerate solutions:

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/GANgearing_5.png" width=500>

#### Parameterization of c.

In practice, we do not backpropagate gradients directly into c.

Instead, we parameterize c as a linear combination of the top-*N* principal directions of W space:

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/GANgearing_6.png" width=200>

Our final GANgealing objective is given by:

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/GANgearing_7.png" width=400>
