# import modules
from arguments import Arguments
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import random
from torch.autograd import Variable
import copy
from torch import nn, optim
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import csv
import time
import math

# main process
args = Arguments()
# define a function for creating datasets
def get_dataset(Centralized=False,unlabeled_data=False):

    transform_train = transforms.Compose([transforms.ToPILImage(),
                                          transforms.RandomCrop(32,mpadding=2),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.491372549, 0.482352941, 0.446666667), (0.247058824, 0.243529412, 0.261568627))])
    
    transform_test = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.491372549, 0.482352941, 0.446666667), (0.247058824, 0.243529412, 0.261568627))])
    
    # prepare train data
    all_trainset = torchvision.datasets.CIFAR10(root="../data",train=True,download=True)
    # prepare test data
    all_testset = torchvision.datasets.CIFAR10(root="../data",train=False,download=True)

    all_train_data = np.array(all_trainset.data)
    all_train_lable = np.array(all_trainset.targets)
    all_test_data = np.array(all_testset.data)
    all_test_label = np.array(all_testset.targets)

    # data heterogenity
    data_proportions = np.random.dirichlet(np.repeat(args.alpha_size,args.worker_num)) 
    train_data_proportions = np.array([0 for _ in range(args.worker_num)])
    test_data_proportions = np.array([0 for _ in range(args.worker_num)])

    for i in range(len(data_proportions)):
        if i==(len(data_proportions)-1):
            train_data_proportions = train_data_proportions.astype("int64")
            test_data_proportions = test_data_proportions.astype("int64")


