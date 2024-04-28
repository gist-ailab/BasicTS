import numpy as np
import torch
import json

import torch

class EarlyStopping:
  """Class to montior the progress of the model and stop early if no improvement on validation set."""

  def __init__(self, ref, patience=7, verbose=False, delta=0):
    """Initializes parameters for EarlyStopping class.

    Args:
      patience: an integer
      verbose: a boolean
      delta: a float
    """
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.delta = delta
    self.ref = ref

  def __call__(self, val_loss, model, path, epoch):
    """Checks if the validation loss is better than the best validation loss.

       If so model is saved.
       If not the EarlyStopping  counter is increased
    Args:
      val_loss: a float representing validation loss
      model: the trained model
      path: a string representing the path to save the model
    """
    score = -val_loss
    if self.best_score is None:
      self.best_score = score
      self.save_checkpoint(epoch, val_loss, model, path)
    elif score < self.best_score + self.delta:
      self.counter += 1
      if self.verbose:
        self.ref.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} [BEST EPOCH: {self.best_epoch}]')
      if self.counter >= self.patience:
        self.early_stop = True
      else:
        self.early_stop = False
    else:
      self.best_score = score
      self.save_checkpoint(epoch, val_loss, model, path)
      self.counter = 0

  def save_checkpoint(self, epoch, val_loss, model, path):
    """Saves the model and updates the best validation loss.

    Args:
      val_loss: a float representing validation loss
      model: the trained model
      path: a string representing the path to save the model
    """
    self.val_loss = val_loss
    if self.verbose:
        self.ref.logger.info(f'Validation loss decreased ({self.ref.target_metric_name}: {self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

    #   print(
    #       f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'

    self.ref.save_best_model(epoch, "val_" + self.ref.target_metric_name, False)

    # torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
    self.val_loss_min = val_loss
    





# class EarlyStopping:
#     def __init__(self, ref, patience=7, verbose=False, delta=0):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.init_metric = -np.Inf
#         self.delta = delta
#         self.ref = ref
#         self.best_epoch = 1

#     def __call__(self, val_loss, model, path, epoch):
#         score = -val_loss
#         self.val_metric = val_loss
#         if self.best_score is None:
#             self.best_score = score
#             self.ref.save_best_model(epoch, "val_" + self.ref.target_metric_name, False)
#             if self.verbose:
#                 self.ref.logger.info(f'Validation loss decreased ({self.ref.target_metric_name}: {self.init_metric:.6f} --> {self.val_metric:.6f}).  Saving model ...')
#             self.best_epoch = epoch
#             self.init_metric = self.val_metric
#                 # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

#             # self.save_checkpoint(val_loss, model, path)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             self.ref.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} [BEST EPOCH: {self.best_epoch}]')
#             # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.ref.save_best_model(epoch, "val_" + self.ref.target_metric_name, False)
#             if self.verbose:
#                 self.ref.logger.info(f'Validation loss decreased ({self.ref.target_metric_name}: {self.init_metric:.6f} --> {self.val_metric:.6f}).  Saving model ...')
#             self.best_epoch = epoch
#             self.init_metric = self.val_metric
#                 # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#             self.counter = 0

    # def save_checkpoint(self, val_loss, model, path):
    #     if self.verbose:
    #         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), path)
    #     self.val_loss_min = val_loss
