import torch
import torch.nn as nn
import sklearn

def get_metrics(targets,predictions):
    return sklearn.metrics.roc_auc_score(targets, predictions)
