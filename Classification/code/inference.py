# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import boto3
import argparse
import json
import logging
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
from monai.config import print_config
from monai.transforms import \
    Compose, LoadImage, Resize, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom
from monai.networks.nets import densenet121

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class DICOMDataset(Dataset):

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = densenet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=3
    ).to(device) 
    
    print("model_dir is", model_dir)
    print("inside model_dir is", os.listdir(model_dir))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        #model.load_state_dict(torch.load(f)) choose the right way to load your model
        model = torch.load(f,map_location=torch.device('cpu') )
    return model.to(device)   


def get_val_data_loader(valX,ValY):
    val_transforms = Compose([
    LoadImage(image_only=True),
    ScaleIntensity(),
    Resize(spatial_size=(512,-1)),
    ToTensor()
    ])
    ValY
    dataset = DICOMDataset(valX,ValY, val_transforms)
    return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)


NUMPY_CONTENT_TYPE = 'application/json'
JSON_CONTENT_TYPE= 'application/json'

s3_client = boto3.client('s3')


def input_fn(serialized_input_data, content_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Received request of type:{content_type}")
    
    print("serialized_input_data is---", serialized_input_data)
    if content_type == 'application/json':
        
        #data = flask.request.data.decode('utf-8')
        data = json.loads(serialized_input_data)
        
        
        bucket=data['bucket']
        image_uri=data['key']
        label=["normal"]
        download_file_name = image_uri.split('/')[-1]
        print ("<<<<download_file_name ", download_file_name)
        ##download from s3 
        print("loaded label is:" , label)
        print("bucket:" , bucket, " key is: ",image_uri," download_file_name is: ",download_file_name)
        
        #check previous 
        
        s3_client.download_file(bucket, image_uri, download_file_name)
        print('Download finished!')
        print('Start to inference for ', download_file_name)
        
        inputs=[download_file_name]
        val_loader = get_val_data_loader(inputs,label)
        print('get_val_data_loader finished!')
        

        for i, val_data in enumerate(val_loader):
            # only have one file each iteraction
            inputs = val_data[0].permute(0,3, 2, 1)
            print("input_fn:",inputs)
            os.remove(download_file_name)
            print('removed the downloaded file!')
            return inputs.to(device)

    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(input_data, model):
    print('Got input Data: {}'.format(input_data))
    print("input_fn in predict:",input_data)
    model.eval()
    response=model(input_data)
    print("response from modeling prediction is", response)
    return response

class_names = [ "Normal","Cap", "Covid"]
def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        print("response in output_fn is", prediction_output)
        pred = torch.nn.functional.softmax(torch.tensor(prediction_output), dim=1)
        print("predicted probability: ", pred)
        
        top_p, top_class = torch.topk(pred, 1)
        x={"results":{"class":top_class.item(), "probability":round(top_p.item(),2), "class":class_names[top_class]}}
        return json.dumps(x)

    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
