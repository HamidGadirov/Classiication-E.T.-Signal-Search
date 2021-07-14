import os
import pathlib
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import codecs

class ModelTrainer:
    def __init__(self,model,fold_idx,dataloaders,criterion,optimizer,metric=None,
                 mode='max',scheduler=None,num_epochs=25,parallel=False,
                 device='cuda:0',save_last_model=True,scheduler_step_per_epoch=True):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.metric = metric
        self.mode = mode
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.parallel = parallel
        self.fold_idx = fold_idx
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_last_model = save_last_model
        self.scheduler_step_per_epoch = scheduler_step_per_epoch
        self.learning_curves = dict()
        self.learning_curves['loss'],self.learning_curves['metric'] = dict(),dict()

        self.learning_curves['loss']['train'],self.learning_curves['loss']['val']=[],[]
        self.learning_curves['metric']['train'],self.learning_curves['metric']['val']=[],[]
        self.best_val_epoch=0
        self.best_val_less = float('inf')
        if self.mode == 'max':
            self.best_val_avg_metric = -float('inf')
        else:
            self.best_val_avg_metric = float('inf')
        self.best_val_metric = 0.0
        self.best_model_wts = None
        self.checkpoint = None

    def train_model(self):
        if self.device.type=='cpu':
            print('Start training the model on CPU')
        elif self.parallel and torch.cuda.device_count()>1:
            print(f'Start training the model on {torch.cuda.device_count()},{torch.cuda.get_device_name(torch.cuda.current_device())} in parallel')
            self.model = torch.nn.DataParallel(self.model)
        else:
            print(f'Start training the model on {torch.cuda.get_device_name(torch.cuda.current_device())}')
        self.model = self.model.to(self.device)
        with codecs.open('log.log','a') as up:
            up.write('\n\n')
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs -1}')
            print('_'*20)
            for phase in ['train','val']:
                if phase=='train':
                    self.model.train()
                else:
                    self.model.eval()
                phase_loss = 0.0
                phase_metric = 0.0
                with torch.set_grad_enabled(phase=='train'):
                    final_targets = []
                    final_outputs = []
                    batch = 0
                    for sample in self.dataloaders[phase]:
                        input,target = sample['image'],sample['target']
                        input,target = input.to(self.device,dtype=torch.float),target.to(self.device,dtype=torch.float)
                        output=self.model(input)
                        loss = self.criterion(output,target.view(-1,1))
#                         print(output)
#                         metric = self.metric(target.detach().cpu().numpy().tolist(),output.detach().cpu().numpy().tolist())
                        phase_loss += loss.item()
#                         phase_metric += metric.item()
                        with np.printoptions(precision=3, suppress=True):
#                             print(f'batch: {batch} batch loss: {loss:.3f} \tmetric: {metric:.3f}')
                            print(f'batch: {batch} batch loss: {loss:.3f} ')

                        del input

                        # Backward pass + optimize only if in training phase:
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                            # zero the parameter gradients:
                            self.optimizer.zero_grad()

                            if self.scheduler and not self.scheduler_step_per_epoch:
                                self.scheduler.step()

                        del loss
                        batch += 1
                        target = target.detach().cpu().numpy().tolist()
                        output = output.detach().cpu().numpy().tolist()
                        final_targets.extend(target)
                        final_outputs.extend(output)
#                         if batch>10:
#                             break
                       
                phase_loss /= len(self.dataloaders[phase])
                phase_metric = self.metric(final_targets,final_outputs)
                self.learning_curves['loss'][phase].append(phase_loss)
                self.learning_curves['metric'][phase].append(phase_metric)

                print(f'{phase.upper()} loss: {phase_loss:.3f} \tavg_metric: {np.mean(phase_metric):.3f}')

                # Save summary if it is the best val results so far:
                if phase == 'val':
                    if self.mode == 'max' and np.mean(phase_metric) > self.best_val_avg_metric:
                        self.best_val_epoch = epoch
                        self.best_val_loss = phase_loss
                        self.best_val_avg_metric = np.mean(phase_metric)
                        self.best_val_metric = phase_metric
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())
                        print(f'best model auc is f{self.best_val_avg_metric}')
                        

                    if self.mode == 'min' and np.mean(phase_metric) < self.best_val_avg_metric:
                        self.best_val_epoch = epoch
                        self.best_val_loss = phase_loss
                        self.best_val_avg_metric = np.mean(phase_metric)
                        self.best_val_metric = phase_metric
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())
                with codecs.open('log.log', 'a') as up:
                    up.write(f"Fold={self.fold_idx}, Epoch={epoch}, Valid ROC AUC={phase_metric}\n")


            # Adjust learning rate after val phase:
            if self.scheduler and self.scheduler_step_per_epoch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(np.mean(phase_metric))
                else:
                    self.scheduler.step()

        if self.save_last_model:
            self.checkpoint = {'model_state_dict': copy.deepcopy(self.model.state_dict()),
                               'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict())}

    def save_results(self, path_to_dir):
        """"
        Save results in a directory. The method must be used after training.
        A short summary is stored in a csv file ('summary.csv'). Weights of the best model are stored in
        'best_model_weights.pt'. A checkpoint of the last epoch is stored in 'last_model_checkpoint.tar'. Two plots
        for the loss function and metric are stored in 'loss_plot.png' and 'metric_plot.png', respectively.
        Parameters
        ----------
        path_to_dir : str
            A path to the directory for storing all results.
        """

        path_to_dir = pathlib.Path(path_to_dir)

        # Check if the directory exists:
        os.makedirs(path_to_dir,exist_ok=True)

        # Write a short summary in a csv file:
        with open(path_to_dir / 'summary.csv', 'w', newline='', encoding='utf-8') as summary:
            summary.write(f'SUMMARY OF THE EXPERIMENT:\n\n')
            summary.write(f'BEST VAL EPOCH: {self.best_val_epoch}\n')
            summary.write(f'BEST VAL LOSS: {self.best_val_loss}\n')
            summary.write(f'BEST VAL AVG metric: {self.best_val_avg_metric}\n')
            summary.write(f'BEST VAL metric: {self.best_val_metric}\n')

        # Save best model weights:
        torch.save(self.best_model_wts, path_to_dir / 'best_model_weights.pt')

        # Save last model weights (checkpoint):
        if self.save_last_model:
            torch.save(self.checkpoint, path_to_dir / 'last_model_checkpoint.tar')

        # Save learning curves as pandas df:
        df_learning_curves = pd.DataFrame.from_dict({
            'loss_train': self.learning_curves['loss']['train'],
            'loss_val': self.learning_curves['loss']['val'],
            'metric_train': self.learning_curves['metric']['train'],
            'metric_val': self.learning_curves['metric']['val']
        })
        df_learning_curves.to_csv(path_to_dir / 'learning_curves.csv', sep=';')

        # Save learning curves' plots in png files:
        # Loss figure:
        plt.figure(figsize=(17.5, 10))
        plt.plot(range(self.num_epochs), self.learning_curves['loss']['train'], label='train')
        plt.plot(range(self.num_epochs), self.learning_curves['loss']['val'], label='val')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=20)
        plt.grid()
        plt.savefig(path_to_dir / 'loss_plot.png', bbox_inches='tight')

        # metric figure:
        train_avg_metric = [np.mean(i) for i in self.learning_curves['metric']['train']]
        val_avg_metric = [np.mean(i) for i in self.learning_curves['metric']['val']]

        plt.figure(figsize=(17.5, 10))
        plt.plot(range(self.num_epochs), train_avg_metric, label='train')
        plt.plot(range(self.num_epochs), val_avg_metric, label='val')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Avg metric', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=20)
        plt.grid()
        plt.savefig(path_to_dir / 'metric_plot.png', bbox_inches='tight')

        print(f'All results have been saved in {path_to_dir}')


