
# NeRF : Representing Scenes as Neural Radiance Fields for View Synthesis

[paper link here](https://arxiv.org/pdf/2003.08934.pdf)

[video](https://www.youtube.com/watch?v=zkeh7Tt9tYQ)

## Abstract

We present a method that achieves SOTA results for synthesizing novel views of complex scenes.

Our algorithm represents a scene using a `fully-connected (non CNN) deep network`, whose input is a single
`continuous 5D coordinate and viewing direction` and whose output is the `volume density and view-dependent emitted
radiance` at that spatial location.

We synthesize views by `querying 5D coordinates along camera rays` and use `classic
volume rendering techniques` to project the output colors and densities into an image.

## Introduction

In this work, we address the long-standing problem of view systhesis in a new way by directly optimizing
parameters of a continuous 5D scene representation to minimize the error of rendering a set of captured images.

Our method optimizes a deep fully connected neural network without any convolutional layers to represent this function
by regressing from a `single 5D coordinate` (*x*, *y*, *z*, θ, Φ) to a `single volume density` and `view-dependent RGB color.`

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/nerf_1.png" width=600>

To render this neural radiance field (NeRF) from a particular viewpoint, we:
* march camera rays through the scene to generate a sampeld set of 3D points.
* use those points and their corresponding 2D viewing directions as input to the neural network to produce an
output set of colors and densities
* use classical volume rendering techniques to accumulate those colors and densities into a 2D image.

> Because this precess is naturally differentiable, we can use gradient descent to optimize this model.

In summary, our technical contributions are:
* An approach for representing continuous scenes with complex geometry and materials as 5D neural radiance fields,
parameterized as basic MLP networks.
* A differentiable rendering precedure based on `classical volume rendering techniques`, which we use to optimize these
representations from standard RGB images.

