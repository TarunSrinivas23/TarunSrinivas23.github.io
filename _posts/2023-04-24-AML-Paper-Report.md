---
layout: post
title:  "Advanced ML: Paper Presentation - Report"
date:   2023-04-24 20:45:00 -0400
categories: jekyll update
---

## 3D Packing for Self-Supervised Monocular Depth Estimation

# Table of Contents
1. [Introduction and Motivation](#introduction-and-motivation)

# Introduction and Motivation

3D Computer Vision is a fast upcoming field with the quick advances seen in Fully Automated Robotics and Self Driving Automobiles. One of the major and important tasks for the **much** sophisticated sensor suite of a robot/automobile would be to see and have a sense of the environment it is in, so that it can avoid collisions, maneuver around and navigate accordingly. The most basic need for any kind of mapping and localization would be to predict depth. Humans use our stereo eyes to have a sense of depth and usually this is acheived using a [LiDAR](https://en.wikipedia.org/wiki/Lidar) in a self-driving car. LiDAR being a common sight in the complex *and costly* sensor suite is one of the most expensive sensors in there. To address this painpoint, researchers have been focusing on trying to get high resolution depth maps and [pointclouds](https://en.wikipedia.org/wiki/Point_cloud) (3d points in a space) from just the usual sensors. And the most commonly used and widely available sensor is, of course, a Camera.

