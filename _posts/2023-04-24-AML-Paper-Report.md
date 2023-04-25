---
layout: post
title:  "Advanced ML: Paper Presentation - Report"
date:   2023-04-24 20:45:00 -0400
categories: jekyll update
---

## 3D Packing for Self-Supervised Monocular Depth Estimation

# Table of Contents
1. [Introduction and Motivation](#introduction-and-motivation)
2. [Literature and Background](#literature-study)
3. [Architecture](#architecture)

# Introduction and Motivation

3D Computer Vision is a fast upcoming field with the quick advances seen in Fully Automated Robotics and Self Driving Automobiles. One of the major and important tasks for the **much** sophisticated sensor suite of a robot/automobile would be to see and have a sense of the environment it is in, so that it can avoid collisions, maneuver around and navigate accordingly. The most basic need for any kind of mapping and localization would be to predict depth. Humans use our stereo eyes to have a sense of depth and usually this is acheived using a [LiDAR](https://en.wikipedia.org/wiki/Lidar) in a self-driving car. LiDAR being a common sight in the complex *and costly* sensor suite is one of the most expensive sensors in there. To address this painpoint, researchers have been focusing on trying to get high resolution depth maps and [pointclouds](https://en.wikipedia.org/wiki/Point_cloud) (3D points in a space) from just the usual sensors. And the most commonly used and widely available sensor is, of course, a Camera.

<img src="{{site.baseurl | prepend: site.url}}media/gifs/packnet-ddad.gif" alt = "This would be the result we expect to see, the pointclouds are then used for further complex tasks in the self driving car. Depth maps play an important role in getting to pointclouds." width="80" height="60" />

The paper proceeds to propose a novel self-supervised deep learning model to do just this. Why self-supervised? Well, collecting labelled data when it comes to self driving applications is a very costly and labourous task, which ofcourse involves a LiDAR which we are trying to replace. The authors propose a U-Net type architecture with some unique modifications, **PackNet**. The authors also claim that PackNet competes and also beats some supervised, unsupervised and other self-supervised monocular models. PackNet is also added with weak supervision from the velocity information available in vehicles. This is proposed by the authors to solve the depth "scale issue" which is rampant in these kinds of monocular applications. As we need stereo information to capture proper scale for our depth predictions, the scale issue comes into the picture which they solve using the velocity information from the images.

<img src="{{site.baseurl | prepend: site.url}}media/images/Picture1.png" alt="A overview of the problem definition. We are trying to get depth maps from a frame from a monocular video feed." />


# Literature Study

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

# Architecture

