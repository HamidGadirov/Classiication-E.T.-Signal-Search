import sys
import argparse
import yaml
import pathlib

import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True
from transform import get_train_transforms, get_valid_transforms
import dataset
import losses
import metrics
import trainer
from network.model import ETNet
import pandas as pd
from utils import util
import sys
sys.path.append('/data/p303872/SET/code/src/network')

def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # read config:
    path_to_data = pathlib.Path(config['path_to_data'])
    path_to_csv = pathlib.Path(config['path_to_csv'])
    path_to_save_dir = pathlib.Path(config['path_to_save_dir'])

    train_batch_size = int(config['train_batch_size'])
    val_batch_size = int(config['val_batch_size'])
    num_workers = int(config['num_workers'])
    lr = float(config['lr'])
    n_epochs = int(config['n_epochs'])
    n_cls = int(config['n_cls'])
    T_0 = int(config['T_0'])
    eta_min = float(config['eta_min'])
    baseline = config['baseline']
    fold_idx = config['fold_idx']
    loss_name = config['loss_name']
    seed = config['seed']
    scheduler_step_per_epoch = config['scheduler_step_per_epoch']
    util.seed_everything(seed)
    # train and val data paths:
    train_images, train_targets, val_images,val_targets= util.get_train_val_paths(all_paths=path_to_data, path_to_train_val_csv=path_to_csv)
    # train and val data transforms:
    # datasets:
    train_set = dataset.ETIDataset(train_images,train_targets,get_train_transforms)
    val_set = dataset.ETIDataset(val_images,val_targets,get_valid_transforms)

    # dataloaders:
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    model=ETNet(baseline,out_dim=n_cls)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    l=config['loss_name']
    criterion = losses.get_loss(loss_name)
        #print('no loss was selected, using dice and focal loss')
    metric = metrics.get_metrics
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)

    trainer_ = trainer.ModelTrainer(
        model=model,
        fold_idx=fold_idx,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        metric=metric,
        scheduler=scheduler,
        num_epochs=n_epochs,
        parallel=False
    )

    trainer_.train_model()
    trainer_.save_results(path_to_dir=path_to_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)
