import numpy as np
import torch
import json


class EarlyStopping:
    def __init__(self, ref, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.ref = ref
        self.best_epoch = 1

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.ref.save_best_model(epoch, "val_" + self.ref.target_metric_name, False)
            if self.verbose:
                self.ref.logger.info(f'Validation loss decreased ({self.ref.target_metric_name}: {self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.best_epoch = epoch
            self.val_loss_min = val_loss
                # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

            # self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.ref.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} [BEST EPOCH: {self.best_epoch}]')
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.ref.save_best_model(epoch, "val_" + self.ref.target_metric_name, False)
            if self.verbose:
                self.ref.logger.info(f'Validation loss decreased ({self.ref.target_metric_name}: {self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.best_epoch = epoch
            self.val_loss_min = val_loss
                # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

            self.counter = 0

    # def save_checkpoint(self, val_loss, model, path):
    #     if self.verbose:
    #         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), path)
    #     self.val_loss_min = val_loss
