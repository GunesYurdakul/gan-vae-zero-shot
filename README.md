# Creativity in Generative Adversarial Networks
### SEMESTER PROJECT REPORT/README
* Güneş Yurdakul - Sciper No: 298878
* Prof. Alexander Alahi
* Advisor: Liu Yuejiang 

1. [ Introduction](#intro)
2. [ Experiments and Usage](#exp)
3. [ References](#ref)

<a name="intro"></a>
## 1. Introduction

Generative models such as GANs[] and Variational Auto-encoders[] have increased the capability of creating models which can generate and manipulate visually realistic images. These models are extremely successful if there exists an excessive amount of training data. The problem is, if the generated samples are able to model the true data distribution. Since GANs are likely suffer from mode collapse, the generations are usually very similar to real images or generators are smoothly interpolating between training samples.  
The aim of our research is explore the creativity of the generated images and finding a way to generate novel and creative images. Since humans can easily generalize from very few examples, we have experimented with various types of GANs and VAEs in zero-shot and few shot settings.
Additionally, we believe that meta-learning, also known as ‘learning to learn’ should also be explored in generative models to improve few-shot generations. Meta-learning models should be able to generalize to new tasks during test time by just a limited number of exposure to a new samples of a new task. A new paper [], presents some promising results of their new model Few-shot Image Generation using Reptile (FIGR). Their work demonstrates the potential of meta-learning to train generative models for few-shot generation and to generate novelty.
We believe that being able to generate images using just a few samples will be very beneficial in other ML related tasks where there is not enough number of training data. For example a dataset containing car accidents will not include many samples in calm neighborhoods. Few shot image generation will be very useful in cases where data augmentation is required.

<a name="exp"></a>
## 2. Experiments and Usage
### 2.1. WHAT WE WANT
images

### 2.2. DesIGN: Design Inspiration from Generative Networks 
We started our research with  DesIGN: Design Inspiration from Generative Networks paper, which is a very recent paper published by Facebook AI Research Team. The aim of the paper is to create creative and visually appealing fashion designs such as bags, t-shirts etc. using Generative Adverserial Networks. They use DCGAN architecture[*] is used and they propose two new loss functions classification loss (LDCLass) and creativity loss(LGCREAt)
*images of functions
Their creativity loss basically computes the Multi Class Cross Entropy(MCE) loss between the class prediction of the dis- criminator and the uniform distribution. The aim is to encourage deviation from existing classes such as shapes and textures in this paper. </br>
#### Toy Dataset Results and Drawbacks
We wanted to observe how their algorithm will work using a toy dataset with 2 features (8 Gaussians). We did some experiments by removing samples of one or more labels. We were curious if the generated samples will be close to the removed samples. However, our results have showed that although some of the generated samples were inbetween gaussian mixtures of given training samples, none of the generated samples were anywhere close to the missing label. 

**** RESULTS ****
#### Required Packages
* Tensorflow []
* Numpy
* Matplotlib

#### Usage
You can run the code using STYLEGAN architecture with 8 Gaussians dataset(default) using the code below: </br >

    $ python3 STYLEGAN.py --fig_name GAN_deneme --label  --notebook --lambda_g 0 --lambda_d 0 --missing_mixt 1
For different settings you can type:
    
    $ python3 STYLEGAN.py --help

### 2.3. CGAN 
### 2.4. INFOGAN
### 2.5. VAE
### 2.6. FIGR
#### Meta-Learning
### Comment on Results

<a name="ref"></a>
## References
[] Sbai, O., Elhoseiny, M., Bordes, A., LeCun, Y., & Couprie, C. (2018). Design: Design inspiration from generative networks. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 0-0).

