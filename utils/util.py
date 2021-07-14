import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
import random
import numpy as np
def get_train_val_paths(all_paths, path_to_train_val_csv):
    df_train=pd.read_csv(path_to_train_val_csv)
    df_train['img_path']=df_train['id'].apply(lambda x:f'../../data/train/{x[0]}/{x}.npy')
    X = df_train.img_path.values
    Y = df_train.target.values
    skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=1024)
    data_index = next(skf.split(X, Y))
    data_index = next(skf.split(X, Y))
    train_index, test_index = data_index[0],data_index[1]

    train_images, valid_images = X[train_index], X[test_index]
    train_targets, valid_targets = Y[train_index], Y[test_index]
    return train_images,train_targets,valid_images,valid_targets
#     for train_index, test_index in skf.split(X, Y):
#             train_images, valid_images = X[train_index], X[test_index]
#             train_targets, valid_targets = Y[train_index], Y[test_index]

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
