
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

> Because this process is naturally differentiable, we can use gradient descent to optimize this model.

In summary, our technical contributions are:
* An approach for representing continuous scenes with complex geometry and materials as 5D neural radiance fields,
parameterized as basic MLP networks.
* A differentiable rendering precedure based on `classical volume rendering techniques`, which we use to optimize these
representations from standard RGB images.
* A `positional encoding` to map each input 5D coordinate into a `higher dimensional space`, which enables us to succesfully
optimize neural radiance fields to represent high-frequency scene content

## Neural Radiance Field Scene Representation
We represent a continuous scene as a 5D vector-valued function wohse input is a 3D location **X** = (*x*, *y*, *z*)
and 2D viewing direction (θ, Φ), and whose output is an emitted color *c* = (*r*, *g*, *b*) and volume density σ.

> *F*<sub>θ</sub> : (**X**, d) -> (*c*, σ)

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/nerf_2.png" width=600>

We encourage the representation to be multiview consistent by `restricting the network` to predict the volume density σ
as a function of `only the location x`, while allowing the RGB color *c* to be `predicted as a function of both location
and viewing direction`.

## Volume Rendering with Radiance Fields

We render the color of any ray passing through the scene using principles from classical volume rendering.

The volume density σ(x) can be interpreted as the `differential probability of a ray terminating` at an
infinitesimal particle at location **X**.

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/nerf_3.png" width=600>

The function T(*t*) denotes the `accumulated transmittance` along the ray from *t*<sub>n</sub> to *t*,
the `probability` that the ray travels from *t*<sub>n</sub> to *t* without hitting any other particle.

Instead using deterministic quadrature, we use a stratified sampling approach where we partition
[*t*<sub>n</sub>, *t*<sub>f</sub>] into *N* evenly-spaced bins and then `draw one sample uniformly at
random from within each bin`:

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/nerf_4.png" width=400>

We use these samples to estimate *C*(r) with the quadrature rule discussed in the volume rendering review by Max:

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/nerf_5.png" width=700>
