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

# define a function for creating datasets
def get_dataset(Centralized=False,unlabeled_data=False):

    transform_train = transforms.Compose([transforms.ToPILImage(),
                                          transforms.RandomCrop(32,mpadding=2),
                                          transforms.ToTensor(),
                                          ])
