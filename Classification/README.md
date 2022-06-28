# Learning Objectives

+ Pre-processing, transformation of DICOM images
+ BYOS model training and evaluation for image classification
+ BYOS model inference for image classification

In this example, we will demonstrate how to integrate the [MONAI](http://monai.io) framework into Amazon SageMaker, with DICOM images as input images and *manifest* json file as labelings.  We will give example code of MONAI pre-processing transforms and neural network (DenseNet) that you can use to train a medical image classification model with DICOM images directly.  Please also visit [Build a medical image analysis pipeline on Amazon SageMaker using the MONAI framework](https://aws.amazon.com/blogs/industries/build-a-medical-image-analysis-pipeline-on-amazon-sagemaker-using-the-monai-framework/) for additional details on how to deploy the MONAI model, pipe input data from S3, and perform batch inferences using SageMaker batch transform.

For more information about the PyTorch in SageMaker, please visit [sagemaker-pytorch-containers](https://github.com/aws/sagemaker-pytorch-containers) and [sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk) github repositories.

# Dataset
The dataset is obtained from this [source COVID-CT-MD](https://github.com/ShahinSHH/COVID-CT-MD), with the total dataset contains volumetric chest CT scans (DICOM files) of 169 patients positive for COVID-19 infection, 60 patients with CAP (Community Acquired Pneumonia), and 76 normal patients. For this demo, only 26 images are selected. 

# Content
+ In `MONAI_BYOS-COVID_2D_demo.ipynb` notebooks, you can see a typical ML  workflow in SageMaker which covers data preparation --> model training --> model deployment --> Inference. 
+ In `Inference-explanation.ipynb` notebook, you can see a few options of deployments, and dive deep into what is happening inside the container. 

```
model.tar.gz/
|- model.pth
|- code/
  |- inference.py
  |- requirements.txt  # only for versions 1.3.1 and higher
```
# Citation of Original Data

[Afshar, Parnian and Heidarian, Shahin and Enshaei, Nastaran and Naderkhani, Farnoosh and Rafiee, Moezedin Javad and Oikonomou, Anastasia and Fard, Faranak Babaki and Samimi, Kaveh and Plataniotis, Konstantinos N and Mohammadi, Arash. "COVID-CT-MD, COVID-19 computed tomography scan dataset applicable in machine learning and deep learning" Scientific Data, 2021](https://diversity.ucsf.edu/resources/unconscious-bias)
