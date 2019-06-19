# Creativity in Generative Adversarial Networks
### SEMESTER PROJECT REPORT/README
* Güneş Yurdakul - Sciper No: 298878
* Prof: Alexander Alahi
* Advisor: Liu Yuejiang 

1. [ Introduction](#intro)
2. [ Experiments and Usage](#exp)
3. [ References](#ref)

<a name="intro"></a>
## Introduction

Generative models such as GANs[] and Variational Auto-encoders[] have increased the capability of creating models which can generate and manipulate visually realistic images. These models are extremely successful if there exists an excessive amount of training data. The problem is, if the generated samples are able to model the true data distribution. Since GANs are likely suffer from mode collapse, the generations are usually very similar to real images or generators are smoothly interpolating between training samples.  
The aim of our research is explore the creativity of the generated images and finding a way to generate novel and creative images. Since humans can easily generalize from very few examples, we have experimented with various types of GANs and VAEs in zero-shot and few shot settings.
Additionally, we believe that meta-learning, also known as ‘learning to learn’ should also be explored in generative models to improve few-shot generations. Meta-learning models should be able to generalize to new tasks during test time by just a limited number of exposure to a new samples of a new task. A new paper [], presents some promising results of their new model Few-shot Image Generation using Reptile (FIGR). Their work demonstrates the potential of meta-learning to train generative models for few-shot generation and to generate novelty.
We believe that being able to generate images using just a few samples will be very beneficial in other ML related tasks where there is not enough number of training data. For example a dataset containing car accidents will not include many samples in calm neighborhoods. Few shot image generation will be very useful in cases where data augmentation is required.

<a name="exp"></a>
## Experiments and Usage

<a name="ref"></a>
## References
