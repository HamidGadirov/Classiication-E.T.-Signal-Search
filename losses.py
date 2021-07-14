import torch
from torch import nn

def get_loss(loss_name='aou'):
    criterion = nn.BCEWithLogitsLoss()
    return criterion