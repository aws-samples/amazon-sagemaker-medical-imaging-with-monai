# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import json
import logging
import os
import shutil
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
from monai.metrics import ROCAUCMetric,get_confusion_matrix,ConfusionMatrixMetric
from monai.config import print_config
from monai.transforms import \
    Compose, LoadImage, Resize, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom, Activations, AddChannel, AsDiscrete, EnsureType
from monai.networks.nets import densenet121
from monai.data import decollate_batch

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


def _get_train_data_loader(batch_size, trainX, trainY, is_distributed, **kwargs):
    logger.info("Get train data loader")
    
    train_transforms = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        RandRotate(range_x=15, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
       # RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        Resize(spatial_size=(512,-1)),
        ToTensor()
    ])

    dataset = DICOMDataset(trainX, trainY, train_transforms)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, **kwargs)

## define data loader for validation dataset
def _get_val_data_loader(batch_size, valX, valY, is_distributed, **kwargs):
    logger.info("Get val data loader")
    
    val_transforms = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        Resize(spatial_size=(512,-1)),
        ToTensor()
    ])


    dataset = DICOMDataset(valX, valY, val_transforms)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=val_sampler is None,
                                       sampler=val_sampler, **kwargs)



## sagemaker training job here
def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    #build file lists
    image_label_list = []
    image_file_list = []
    metadata = args.data_dir+'/manifest.json'   
    # Load Labels
    # open json files
    f1_metric = ConfusionMatrixMetric(metric_name='f1 score')
    with open(metadata) as f:
        manifest = json.load(f)
    
    
    my_dictionary = {'cap':1, 'normal':0, 'covid':2}
    class_names = list(my_dictionary.keys())
    num_class = len(class_names)
    
    #y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
    y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)])
    y_pred_trans = Compose([EnsureType(),  AsDiscrete(to_onehot=num_class)])  
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.debug('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        
    
    
    
    for file in manifest:
            name = file['filename']
            filename = args.data_dir+'/'+name
            image_file_list.append(filename)
            label=file['content']['label']
            label_numeric=my_dictionary[label]
            image_label_list.extend([[label_numeric]])
    #print('image_label_list ---', image_label_list)
    
    print("Training count =",len(image_file_list))
    
    # divide the data into training and validation dataset
    val_frac = 0.2
    length = len(image_file_list)
    indices = np.arange(length)
    np.random.shuffle(indices)
    val_split = int(val_frac * length)
    val_indices = indices[:val_split+3]
    train_indices = indices[val_split:]
    print("length of validation dataset:", len(val_indices))
    
    train_x = [image_file_list[i] for i in train_indices]
    train_y = [image_label_list[i] for i in train_indices]
    val_x = [image_file_list[i] for i in val_indices]
    val_y = [image_label_list[i] for i in val_indices]
    
    ## get data loader for training dataset and validation dataset
    train_loader = _get_train_data_loader(args.batch_size, train_x, train_y, False, **kwargs)
    val_loader = _get_val_data_loader(args.batch_size, val_x, val_y, False, **kwargs)

    #create model
    model = densenet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=3
    ).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    epoch_num = args.epochs
    val_interval = 1

    #train model
    best_f1=-1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    val_interval =1
    for epoch in range(epoch_num):
        logger.info('-' * 10)
        logger.info(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
           
            step += 1
            #inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            inputs = batch_data[0].to(device)
            
            #logger.info('label type: ', batch_data[1].type())
            print('inputs shape is -----',inputs.shape)
            #inputs = inputs.to(memory_format=torch.channels_last) ######### debug here
            inputs = inputs.permute(0,3, 2, 1)
            print('inputs shape after is -----',inputs.shape)

            # print('label from batch is',batch_data[1][0].to(device))  
            #x=[[el] for el in batch_data[1]]
            #label  = torch.tensor(batch_data[1])
            labels = batch_data[1][0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            logger.info(f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
            epoch_len = len(train_loader.dataset) // train_loader.batch_size        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0: ## evaluate the model with validation dataset
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)##initiate the predicted array
                y = torch.tensor([], dtype=torch.long, device=device)

                print('inputs shape after is -----',inputs.shape)

        # print('label from batch is',batch_data[1][0].to(device))  
        #x=[[el] for el in batch_data[1]]
        #label  = torch.tensor(batch_data[1])
        #labels = batch_data[1].to(device)

            for val_data in val_loader:
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1][0].to(device),
                )
                val_images = val_images.permute(0,3, 2, 1) ## permuate the image, channel first
                
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                
                print("y_pred is:", y_pred, "actual y is : ", y)
                
            
            
            y_pred2=torch.nn.functional.softmax(y_pred, dim=1)
            top_p, top_class=torch.topk(y_pred2,1)
            y_pred_class=[x.item() for x in top_class]
            
            y_onehot = [y_trans(i) for i in decollate_batch(y)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(top_class)]
             
            print("ground truth is : ",  y)
            print("predicted truth is : ",  y_pred_class)
            
            f1_metric(y_onehot, y_pred_act)
            result = f1_metric.aggregate()
            print("~~~results after f1 is:", result)
            f1_metric_numpy=result[0].numpy()[0]
            
      
            if(f1_metric_numpy>best_f1):
                print("saved new best metric model at epoch:", epoch + 1,  "f1 score is:",f1_metric_numpy )
                best_f1 = f1_metric_numpy
                best_metric_epoch = epoch + 1
                save_model(model, args.model_dir)
            f1_metric.reset()
            
            
            ## second metrics: on accuracy 
            acc_value = torch.eq(torch.FloatTensor(y_pred_class), y)
            
            acc_metric = acc_value.sum().item() / len(acc_value)
            metric_values.append(acc_metric)
       
            if((acc_metric >= best_metric) ):#|((result > best_metric-0.02) & (epoch>(10+best_metric_epoch)))):
                print("new best metric model at epoch:", epoch + 1)
                best_metric = acc_metric
                best_metric_epoch = epoch + 1
                #save_model(model, args.model_dir)

                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current AUC: {acc_metric:.4f}"
                f" current average accuracy: {acc_metric:.4f}"
                f" best AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

    # model code directory
    #save_model(model, args.model_dir)
    model_code_dir = os.path.join(args.model_dir, 'code') 
    os.makedirs(model_code_dir)
    shutil.copy('/opt/ml/code/inference.py', model_code_dir) ## copy the inference file
    shutil.copy('/opt/ml/code/requirements.txt', model_code_dir) # copy requirement.txt
    return 


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    #torch.save(model.cpu().state_dict(), path)
    torch.save(model, path) ## save a model artifact in the speficied folder


    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    
    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())


