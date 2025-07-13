# Estimating Tumor Size on 2D Images of Non-Uniform Body Cavity with Object Detection and Depth Analysis

Source code implementations to support experiments reproducibility

## Prerequisites
* Python 3.10
* Ubuntu 20.04 LTS

## Structure
This repository consists of three directories:
* Codes
* Python Notebooks
* Data schema

## Walkthrough
<img width="995" height="322" alt="image" src="https://github.com/user-attachments/assets/ce8909fb-a9e5-451c-8eec-bcea8706c1d7" />

There are three core processes within the experiments:
1. Tumor detection model training with YOLOv11
2. Estimating depth with Depth Anything V2
3. Image stitching from videos

## References
* [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
* [YOLOv11](https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolo11-object-detection-on-custom-dataset.ipynb)
* [Image Stitching](https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html)
