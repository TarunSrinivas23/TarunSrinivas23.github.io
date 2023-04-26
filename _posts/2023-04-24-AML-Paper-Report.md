---
layout: post
title:  "Advanced ML: Paper Presentation - Report"
date:   2023-04-24 20:45:00 -0400
categories: jekyll update
---

### 3D Packing for Self-Supervised Monocular Depth Estimation

## Table of Contents
1. [Introduction and Motivation](#introduction-and-motivation)
2. [Literature and Background](#literature-study)
3. [Architecture](#architecture)
4. [Experiments](#experiments)

    - [PackNet](#packnet)
    - [Packing and Unpacking](#packing-and-unpacking-blocks)
5. [Algorithm](#algorithm)

    - [Pointcloud Reprojection](#pointcloud-reprojection)
    - [PoseNet](#posenet)
    - [View Synthesis](#view-synthesis)
    - [Loss Function](#loss-function)
6. [Evaluation and Results](#evaluation-and-results)
7. [Conclusions and Future Work](#conclusions)

Paper link : [https://arxiv.org/pdf/1905.02693.pdf](https://arxiv.org/pdf/1905.02693.pdf)
## Introduction and Motivation

3D Computer Vision is a fast upcoming field with the quick advances seen in Fully Automated Robotics and Self Driving Automobiles. One of the major and important tasks for the **much** sophisticated sensor suite of a robot/automobile would be to see and have a sense of the environment it is in, so that it can avoid collisions, maneuver around and navigate accordingly. The most basic need for any kind of mapping and localization would be to predict depth. Humans use our stereo eyes to have a sense of depth and usually this is acheived using a [LiDAR](https://en.wikipedia.org/wiki/Lidar) in a self-driving car. LiDAR being a common sight in the complex *and costly* sensor suite is one of the most expensive sensors in there. To address this painpoint, researchers have been focusing on trying to get high resolution depth maps and [pointclouds](https://en.wikipedia.org/wiki/Point_cloud) (3D points in a space) from just the usual sensors. And the most commonly used and widely available sensor is, of course, a Camera.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/gifs/packnet-ddad.gif"  alt= "Error lol. I cant really help you" title = "This would be the result we expect to see, the pointclouds are then used for further complex tasks in the self driving car. Depth maps play an important role in getting to pointclouds." width="640" height="480" />

</p>

The paper proceeds to propose a novel self-supervised deep learning model to do just this. Why self-supervised? Well, collecting labelled data when it comes to self driving applications is a very costly and labourous task, which ofcourse involves a LiDAR which we are trying to replace. The authors propose a U-Net type architecture with some unique modifications, **PackNet**. The authors also claim that PackNet competes and also beats some supervised, unsupervised and other self-supervised monocular models. PackNet is also added with weak supervision from the velocity information available in vehicles. This is proposed by the authors to solve the depth "scale issue" which is rampant in these kinds of monocular applications. As we need stereo information to capture proper scale for our depth predictions, the scale issue comes into the picture which they solve using the velocity information from the images.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture1.png" alt= "Error lol. I cant really help you" title="A overview of the problem definition. We are trying to get depth maps from a frame from a monocular video feed." />

</p>

## Literature Study

I found a bunch of papers doing monocular self supervised depth prediction for self driving cars online, but only a few were most relevant to understand what was going on in this particular paper.

+ [Unsupervised learning of Depth and Ego-Motion from Video - Tinghui Zhou,
Matthew Brown et al.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf)

    This paper introduces the core architecture that our paper follows, they use previous frames from the video to produce priors that can be used to do unsupervised training. They also propose a CNN architecture, PoseNet, which regresses the ego-motion parameters for the previous frames to the current frame. The authors use this information to reconstruct the current frame using the previous frames and proceed with unsupervised training which we can understand in depth later.

    Key takeaways would be:

    - A good understanding of the architecture that is generally used for these kinds of applications.
    - An understanding of the "view-synthesis" part which is a major portion of this paper.
    - PoseNet and it's functioning as it is used in our paper.

+ [Digging Into Self-Supervised Monocular Depth Estimation - Clément Godard et al.](https://arxiv.org/pdf/1806.01260.pdf)

    This paper was also useful for me as it sheds light on the loss function and it's different parts and how they help the model learn better features in a monocular self-supervised network.

    Key takeaways would be:

    - A good understanding of the different parts of the loss function.
    - Good explanations on the auto masking featuring in the objective function in our paper.

+ [Image quality assessment: from error visibility to structural similarity - Zhou Wang et al.](https://ieeexplore.ieee.org/document/1284395)

    I found this paper useful especially in understanding the Structural similarity term in our objective function. We will need this to understand our main photometric loss function.

    Key Takeaways:

    - A good understanding on Structural similarity and how it helps improving photometric loss.

## Architecture

The architecture that the authors propose is not very different from any other self supervised monocular depth prediction model, it follows the same basic skeleton. The image below shows the architecture that the authors propose, with PackNet depth prediction highlighted.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture6.png" alt= "Error lol. I cant really help you" title= "Architecture overview for the proposed self-supervised network." />

</p>

The $$I_t$$ here is the target image that we wish to predict depth for. And this is passed in through the model and we get an output from PackNet, $$\hat D_t$$. $$I_s$$ is the context frames, could be previous frames in our case. $$\hat I_t$$ here is the "synthesised" target image which is the output from the view synthesis block. $$\hat I_t$$ and $$I_t$$ are then used to model the photometric loss to backpropogate and train the model. This is how they propose to avoid labelled data for training. We will see this in much more depth further down the blog though.

## Experiments

# PackNet

The authors introduce PackNet, which is basically just a U-Net, which is very common in vision research, mostly seen in segmentation tasks where we need an image as input and output for the model. You can refer to different papers, but [Olaf Ronneberger et al.](https://arxiv.org/pdf/1505.04597.pdf) introduced this type of network for image segmentation tasks.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture7.png" alt= "Error lol. I cant really help you" title= "PackNet model summary." />

</p>

The authors are also introducing a new block in this encoder decoder architecture, which we will look at next, then proceed to look at the other blocks in the architecture used for monocular depth estimation.

# Packing and Unpacking blocks

The authors proceed to introduce their modification to the traditional U-Net that they felt would imporve this model for our application. They propose to use new downsampling and upsampling blocks, which are common in U-Net architectures to extract features and predict features.

The traditional way this is carried out is using "Max-Pooling" and "Bi-Linear Upsampling. For reference, you can see this toy U-Net architecture where the red arrows indicate max-pooling to downsample and the green arrows indicate bi-linear upsampling, to revert to higher feature spaces. Image was grabbed from [Olaf Ronneberger et al.](https://arxiv.org/pdf/1505.04597.pdf).


<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture8.PNG" alt= "Error lol. I cant really help you" title= "Toy U-Net with Max-Pooling (red) and BiLinear Upsampling (green)" />

</p>

But this process is notorious for losing information from the feature maps. Since max pooling works as a dimension redux, it cannot retain information which might be useful for depth prediction tasks. The intuition behind max pooling is shown below, notice that we involve a max operator, which is not learnable and also chooses to reduce dimensionality in a non linear way.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture9.png" alt= "Error lol. I cant really help you" title= "BTS- The Max-Pooling layer." />

</p>

This is where Packing and Unpacking comes into the picture, the authors choose these layers as they can do exactly this, i.e, reduce dimensionality of the feature map, but these layers can do it without losing much information. The way it achieves this is by using different 3D convolution operations to downsize the feature maps so that the network can **LEARN** how to advantageously downsample without being forced to lose information. The inverse is true for upsampling. An overview of the layer and it's different components is shown below. Notice how the dimensions of a toy input changes. In the end we end up with smaller feature maps but with **learned** downsampling.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture10.png" alt= "Error lol. I cant really help you" title= "Packing and Unpacking Layers with their effect on feature dimensions." />

</p>

We can see that this type of downsampling helps the network from the results the authors published but to gather more understanding of how it retains information, I found a graphic in the authors paper that explained it better, which is attached below. We can see that using the conventional way of downsampling and upsampling, we can only recontruct a *blurry* version of the original image, whereas packing and unpacking the image gave out a much more *crisp* output. These kind of details go a long way in predicting depth.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture11.png" alt= "Error lol. I cant really help you" title= "Packing and Unpacking Layers vs Conventional Methods." />

</p>

## Algorithm

# PointCloud Reprojection

Now that we have PackNet out of the way, say we have a depth map prediction. We get to the point where we need evaluate this depth map and optimize something to make future predictions better. First step in this would be PointCloud reprojection.

In easy terms, reprojection takes an image, it's corresponding depth and camera parameters as input and projects said image into a 3D space. This outputs a pointcloud which we can view and show off to our peers ;)

Using traditional computer vision techniques, we can reproject pointclouds onto a 3D space, with certain rules.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture12.PNG" alt= "Error lol. I cant really help you" title= "PointCloud Reprojection" />

</p>

This pointcloud is then passed on to a block called "**View-Synthesis**" which puts together the priors and sets up self-supervised learning. Another important aspect of View-Synthesis is **PoseNet**

# PoseNet

Predicts the transformation matrix between two context frames in the video. The authors use the architecture proposed by [Tinghui Zhou,
Matthew Brown et al.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf) . It is a 7-layer convolutional network, ending with a 1x1 convolution layer with a regression output.

The input is the target image ($$I_t$$) and the context images ($$I_S$$). The expected output is one or more 6 DoF arrays, which represents the ego-motion (Transformation matrices - for all the robotics geeks out there) from each context image to the target image.

# View Synthesis

Now that we have all the ingredients, we proceed to view synthesis and the loss function for training.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture13.png" alt= "Error lol. I cant really help you" title= "View Synthesis" />

</p>

View synthesis basically takes the depth map from PackNet, the target image, context frames, their respective transformations from PoseNet and camera parameters as input.

Henceforth what view-synth is trying to do, is reconstruct the target image using the context images. This is done by moving the context frame to the target frame using the output $$T$$ from PoseNet, then sample corresponding points between the target and context frames from the context frame.

Corresponding points are found by projecting a coordinate from target frame to a 3D space using the depth values from $$\hat D_t$$ and reprojecting it back onto the context frame. This would give us it's corresponding pixel location in the context frames which is known. So now we just sample the rgb values from the context frames and recontruct the target frame. By doing this, we introduce geometric contraints on the predictions, which introduces the prior that is needed to start self supervised learning. The graphic above tries to explain exactly this.

This type of view synthesis was proposed and well explained by [Tinghui Zhou,
Matthew Brown et al.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf). Going through this paper also helped me understand what was actually going on.

# Loss Function

Another important aspect of the paper was the loss function. Authors propose a similar approach as in [Clément Godard et al.](https://arxiv.org/pdf/1806.01260.pdf) but add on different modifications to favour the application more. Let's start off with the objective function as a whole and break it down to understand each part.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture22.png" alt= "Error lol. I cant really help you" title= "Photometric loss" width = 800 height = 80 />

</p>


<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture15.png" alt= "Error lol. I cant really help you" title= "Photometric loss" />

</p>

The authors define $$L_p$$ as the photometric loss term,

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture16.png" alt= "Error lol. I cant really help you" title= "Photometric loss" width = 800 height = 80 />

</p>

The photometric term is shown to be a combination of an L1 loss term and a Structural Similarity term (SSIM), which was proposed in [Zhou Wang et al.](https://ieeexplore.ieee.org/document/1284395). Structural similarity had a simple explanation in the [Wiki](https://en.wikipedia.org/wiki/Structural_similarity) too, for your reference.

Now we minimize this loss across all context frames. The intuition behind this is to avoid occulded pixels in the context frames affecting loss term in a huge way. The reason being occluded pixels are not a mistake, but just a consequence. This was proposed in [Clément Godard et al.](https://arxiv.org/pdf/1806.01260.pdf) for reference. 
<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture17.png" alt= "Error lol. I cant really help you" title= "Per Pixel Minima" />

</p>

The objective function is also a combination of different loss terms, masks and regularization terms:

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture22.png" alt= "Error lol. I cant really help you" title= "Photometric loss" width = 800 height = 80 />

</p>


<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture18.png" alt= "Error lol. I cant really help you" title= "Auto Masking" width = 800 height = 80 />

</p>

Intuition behind this mask ($$M_p$$) is, if the minimum loss for the warped image itself is more than the loss from the unwarped source image with respect to the target frame, it means that that specific pixel is static. And these pixels are masked out.

$$M_t$$ is a binary mask that masks out any depth values that are out of bounds. Generally, we have a range within which the model is expected to predict.

The final term is regularization term to avoid very high values of depth in the predictions.

This is the essense of the training procedure and experiments for this paper. The loss function is then used to penalize the model and backpropogate and train.

## Evaluation and Results

The authors use common evaluation metrics used by other monocular depth prediction papers, to evaluate their model. A consice explanation of each of these metrics can be found in [David Eigen et al.](https://arxiv.org/abs/1406.2283). I also attach an image I found to be helpful below

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture19.png" alt= "Error lol. I cant really help you" title= "Evaluation Metrics" />

</p>

The authors evaluate their model on the [KITTI](https://www.cvlibs.net/datasets/kitti/) benchmark dataset which is a de-facto dataset in computer vision applications.

Experiments were conducted on different resolutions, different training datasets (KITTI and a combination of CityScapes and KITTI). The evaluation being done on strictly KITTI dataset. We can expect results to be somewhat similar to the authors given below.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture20.png" alt= "Error lol. I cant really help you" title= "Results - Quantitative" />

</p>

The table shows the models performance across multiple evaluation metrics (lower is better for all, except the thresholds). As we can see the model outperforms other unsupervised methods and self supervised methods with good margins. So the *Qualitative* performance does not fall far behind.

<p style="text-align: center;">

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture21.png" alt= "Error lol. I cant really help you" title= "Results - Qualitative" />

</p>

I found much more detail in results, which I feel was due to the lossless downsampling that the authors introduced.

## Conclusions

The model seems to outperform other monocular depth models in the KITTI dataset, authors also provide results from other datasets in their paper [here](https://arxiv.org/pdf/1905.02693.pdf).

Authors claim that although purely trained on unlabeled monocular videos, their approach outperforms other existing self and semi-supervised methods and is even competitive with fully-supervised methods while able to run in real-time. Which is very important when we think about a self driving car scenario.

The GitHub repository [here](https://github.com/TRI-ML/packnet-sfm) released contains the code to implement the model along with intructions to implement it.

# Future Work

As I read, understood an dimplemented the paper, I found that monocular computer vision has come a long way powered by advances in deep learning, computer vision and machine learning. But I came across some shortcomings too, as all monocular papers try to address or compromise on, scale issues are rampant as we cannot possibly predict accurate scale with just a single image input. There is active research on solving this scale issue when using monocular input, which I found very interesting.

## References

1. [3D Packing for Self-Supervised Monocular Depth Estimation, Vitor Guizilini et al.](https://arxiv.org/pdf/1905.02693.pdf)
2. [Unsupervised learning of Depth and Ego-Motion from Video - Tinghui Zhou,
Matthew Brown et al.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf)
3. [Digging Into Self-Supervised Monocular Depth Estimation - Clément Godard et al.](https://arxiv.org/pdf/1806.01260.pdf)
4. [Image quality assessment: from error visibility to structural similarity - Zhou Wang et al.](https://ieeexplore.ieee.org/document/1284395)
5. [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, David Eigen et al.](https://arxiv.org/abs/1406.2283)
