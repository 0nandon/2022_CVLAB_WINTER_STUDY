
# Unpaired Image-to-Image Translation using Cycle-consistent Adversial Networks

[paper link here](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)

## Abstract

Our goal is to `learn a mapping *G : X = Y*` such that the distribution of images from
*G(X)* is indistinguishable from the distribution *Y* using an adversarial loss.

Qualitative results are presented on several tasks where `paired training data does not exist.`

## Introduction

In this paper, we present a system that can learn to do the same:
capturing special characteristics of one image collection and figuring out how
these chracteristics could be translated into the other image collection,
all `in the absence of any paired training examples.`

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/Imagegeneration_3_1.png" width=500>

Because obtaining paired training data can be difficult and expensive,
we therefore seek an algorithm that can learn to translate between domains
`without paired input-output examples.`

We therefore seek an algorithm that can learn to translate between domains
`without paired input-output examples.` 

We may train mapping *G : X = Y* such that the output *y<sup>~</sup> = G(x), x ∈ X* is indistinguishable
from images *y ∈ Y* by an adversary trained to classify *y<sup>~</sup>* apart from *y*.

However, such a translation does not guarantee that the individual inputs and outputs *x* and *y* are paired up
in a meaningful way - there are infinitely many mappings *G* that will induce the same distribution over *y<sup>~</sup>*.

> In practice, we have found it difficult to optimize the adversial objective in isolation: standard procedures often lead
> to the well-known problem of mode collapse.

Therefore, we exploit the property that translation should be `'cycle consistent'.`

Matematically, if we have a translator *G : X = Y* and another translator *F : Y = X*, then
*G* and *F* should be inverses of each other. We apply this structural assumption by training both
the mapping *G* and *F* simultaneously, and adding a *cycle consistent loss*.

We apply our method to a wide range of applications below.
* style transfer
* object transfiguration
* attribute transfer
* photo enhancement

## Formulation

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/Imagegeneration_3_2.png" width=900>

## Imlementation

We apply two techniques from recent works to stabilize our model training procedure.

* L<sub>GAN</sub>, we replace the negative log likelihood objective by a least square loss.
* To reduce model oscillation, we update the discrimators using a history of generated images rather than
the ones produced by the latest generative images.

## Results

We study the importance of both the adversarial loss and the cycle consistent loss,
and compare our full method agaist several variants.

We compare our method against several baselines below.
* CoCAN
* Pixel loss + GAN
* Feature loss + GAN
* BiGAN/ALI
* pix2pix

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/Imagegeneration_3_4.png" width=500>



## Limitations and Discussion

* Some failures occur in extreme transformations such as geometric changes.
* Some failure cases are caused by the distribution characteristic of the training datasets.

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/Imagegeneration_3_3.png" width=500>

> Figure 12 has completely failed, because our model was trained in the wild horse, zebra synsets of ImageNet,
> which does not contain images of a person riding horse or zebra.

* We also observe a lingering gap between the results achievable with paired training data and those achieved
by our unpaired method.
