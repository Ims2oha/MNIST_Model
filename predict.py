import torch
import torch.nn

import sys
import numpy as np
import matplotlib as plt

from model import ImageClassifier

from untils import load_mnist
from untils import split_data
from untils import get_hidden_sizes

model_fn = "./model.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load(fn, device):
    d = torch.load(fn, map_location=device)

    return d['model'], d['config']

def plot(x, y_hat):
    for i in range(x.size(0)):
        