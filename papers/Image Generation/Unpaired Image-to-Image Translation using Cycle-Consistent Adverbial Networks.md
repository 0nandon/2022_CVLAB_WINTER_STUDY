
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

## Formulation

<img src="https://github.com/0nandon/2022_CVLAB_WINTER_STUDY/blob/main/photo/Imagegeneration_3_2.png" width=700>
