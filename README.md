# MONAI-MedicalImage-SageMaker

This repository contains examples and related resources showing you how to you can use deep learning algorithms from MONAI libries to train models and conduct inference using Amazon SageMaker with Bring Your Own Script (BROS) model. 


 
  
# Overview
We demo the capability of using MONAI in Amazon SageMaker to transform, train the images and then annotate the medical images automatically. This repo was adapted from local model of 2 examples from [MONAI project](https://github.com/Project-MONAI), specifically on the [2D classification](https://github.com/Project-MONAI/tutorials/tree/main/2d_classification) and [3D segmentation](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb) from [MONAI tutorials](https://github.com/Project-MONAI/tutorials). 

# Repository Structure

+ Classification
  Classification of 2D CT images in DICOM format into  COVID-19 infection, Community Acquired Pneumonia and  normal conditions. 
  
+ Segementation 
  Segmentation of spleen from 3D CT images, which are captured in NIFTI format.  
# License
This library is licensed under the MIT-0 License. See the LICENSE file.
