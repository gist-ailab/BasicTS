import torch

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner
from basicts.utils.early_stopping import EarlyStopping
from easytorch.utils import TimePredictor, get_local_rank
from easytorch.utils.data_prefetcher import DevicePrefetcher
import time
from tqdm import tqdm
from easytorch.core.checkpoint import save_ckpt, clear_ckpt, backup_last_ckpt, load_ckpt
from torch.nn.parallel import DistributedDataParallel as DDP
import os 

class SimpleLongTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):
    """Simple Runner: select forward features and target features. This runner can cover most cases."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.target_metric_name = cfg.get("TARGET_METRICS", "MSE")
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def on_epoch_end(self, epoch: int):
        """Callback at the end of an epoch.

        Args:
            epoch (int): current epoch.
        """

        # print train meters
        self.print_epoch_meters('train')
        # tensorboard plt meters
        self.plt_epoch_meters('train', epoch)
        # validate
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        
        # test
        if self.test_data_loader is not None and epoch % self.test_interval == 0:
            self.test_process(train_epoch=epoch)   
            
        # save model
        self.save_model(epoch)
          
    def save_model(self, epoch: int):
        """Save checkpoint every epoch.

        checkpoint format is {
            'epoch': current epoch ([1, num_epochs]),
            'model_state_dict': state_dict of model,
            'optim_state_dict': state_dict of optimizer
        }

        Decide whether to delete the last checkpoint by the checkpoint save strategy.

        Args:
            epoch (int): current epoch.
        """

        model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'best_metrics': self.best_metrics,
            'early_stop_counter': self.early_stopping.counter,
            'early_stop_val_loss': self.early_stopping.val_metric
        }


        # backup last epoch
        last_ckpt_path = self.get_ckpt_path(epoch - 1)
        backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)

        # save ckpt
        ckpt_path = self.get_ckpt_path(epoch)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

        # clear ckpt every 10 epoch or in the end
        if epoch % 10 == 0 or epoch == self.num_epochs:
            clear_ckpt(self.ckpt_save_dir)    
            
    def train(self, cfg):
        """Train model.

        Train process:
        [init_training]
        for in train_epoch
            [on_epoch_start]
            for in train iters
                [train_iters]
            [on_epoch_end] ------> Epoch Val: val every n epoch
                                    [on_validating_start]
                                    for in val iters
                                        val iter
                                    [on_validating_end]
            if validation loss is not improved:
                break (early stopping)
        [on_training_end]

        Args:
            cfg (Dict): config
            """

        self.init_training(cfg)

        # train time predictor
        train_time_predictor = TimePredictor(self.start_epoch, self.num_epochs)
        self.early_stopping = EarlyStopping(self, patience=cfg.get('TRAIN.EARLY_STOPPING.PATIENCE', 10), verbose=True)
        # training loop
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1
            self.on_epoch_start(epoch)
            epoch_start_time = time.time()
            # start training
            self.model.train()

            # tqdm process bar
            if cfg.get('TRAIN.DATA.DEVICE_PREFETCH', False):
                data_loader = DevicePrefetcher(self.train_data_loader)
            else:
                data_loader = self.train_data_loader
            data_loader = tqdm(data_loader) if get_local_rank() == 0 else data_loader

            # data loop
            for iter_index, data in enumerate(data_loader):
                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
            # update lr_scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = time.time()
            # epoch time
            self.update_epoch_meter('train_time', epoch_end_time - epoch_start_time)
            self.on_epoch_end(epoch)

            # self.early_stopping(self.epoch_meters['val_loss'].avg, self.model, self.checkpoint_path)
            #print("dddd: ", self.meter_pool.get_avg(f'val_{self.target_metric_name}'))
            self.early_stopping(self.meter_pool.get_avg(f'val_{self.target_metric_name}'), self.model, self.get_ckpt_path(epoch), epoch)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
        
            # reset meters
            self.reset_epoch_meters() 
            expected_end_time = train_time_predictor.get_expected_end_time(epoch)

            # estimate training finish time
            if epoch < self.num_epochs:
                self.logger.info('The estimated training finish time is {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_end_time))))

        # log training finish time
        self.logger.info('The training finished at {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        ))

        self.on_training_end()  

    # @master_only
    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        """Save the best model while training.

        Examples:
            >>> def on_validating_end(self, train_epoch: Optional[int]):
            >>>     if train_epoch is not None:
            >>>         self.save_best_model(train_epoch, 'val/loss', greater_best=False)

        Args:
            epoch (int): current epoch.
            metric_name (str): metric name used to measure the model, must be registered in `epoch_meter`.
            greater_best (bool, optional): `True` means greater value is best, such as `acc`
                `False` means lower value is best, such as `loss`. Defaults to True.
        """

        metric = self.meter_pool.get_avg(metric_name)
        best_metric = self.best_metrics.get(metric_name)
        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            model = self.model.module if isinstance(self.model, DDP) else self.model
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'best_metrics': self.best_metrics,
                'early_stop_counter': self.early_stopping.counter,
                'early_stop_val_loss': self.early_stopping.val_loss
            }
            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                '{}_best_{}.pt'.format(self.model_name, metric_name.replace('/', '_'))
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)


    def load_model_resume(self, strict: bool = True):
        """Load last checkpoint in checkpoint save dir to resume training.

        Load model state dict.
        Load optimizer state dict.
        Load start epoch and set it to lr_scheduler.

        Args:
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        """

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            self.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_epoch = checkpoint_dict['epoch']
            if checkpoint_dict.get('best_metrics') is not None:
                self.best_metrics = checkpoint_dict['best_metrics']
            if self.scheduler is not None:
                self.scheduler.last_epoch = checkpoint_dict['epoch']
            self.logger.info('Resume training')
        
            if checkpoint_dict['early_stop_counter']>= self.early_stopping.patience:
                RuntimeError("Early stopping")
            self.early_stopping.counter = checkpoint_dict['early_stop_counter']
            self.early_stopping.loss = checkpoint_dict['early_stop_val_loss']

        except (IndexError, OSError, KeyError):
            pass

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C]

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target feature.

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history ata).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            dict: keys that must be included: inputs, prediction, target
        """

        # preprocess
        future_data, history_data = data
        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape
        

        history_data = self.select_input_features(history_data)
        if train:
            future_data_4_dec = self.select_input_features(future_data)
        else:
            future_data_4_dec = self.select_input_features(future_data)
            # only use the temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # model forward
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec, batch_seen=iter_num, epoch=epoch, train=train)
    

        # parse model return
        if isinstance(model_return, torch.Tensor): model_return = {"prediction": model_return}
        if "inputs" not in model_return: model_return["inputs"] = self.select_target_features(history_data)
        if "target" not in model_return: model_return["target"] = self.select_target_features(future_data)
        # print(list(model_return["prediction"].shape)[:3])
        assert list(model_return["prediction"].shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        return model_return
