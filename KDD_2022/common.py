import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from model_rnn import BaselineRNNModel
from model_gru import BaselineGRUModel
from model_lstm import BaselineLSTMModel
from prepare import prep_env
from wind_turbine_data import BaseWPFDataset
from test_data import TestData

def adjust_learning_rate(optimizer, epoch, args):
    # type: (paddle.optimizer.Adam, int, dict) -> None
    """
    Desc:
        Adjust learning rate
    Args:
        optimizer:
        epoch:
        args:
    Returns:
        None
    """
    # lr = args.lr * (0.2 ** (epoch // 2))
    lr_adjust = {}
    if args["lr_adjust"] == 'type1':
        # learning_rate = 0.5^{epoch-1}
        lr_adjust = {epoch: args["lr"] * (0.50 ** (epoch - 1))}
    elif args["lr_adjust"] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        optimizer.set_lr(lr)


class EarlyStopping(object):
    """
    Desc:
        EarlyStopping
    """
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = False

    def save_checkpoint(self, val_loss, model, path, tid, log_path):
        # type: (nn.MSELoss, BaselineGruModel, str, int, str) -> None
        """
        Desc:
            Save current checkpoint
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        self.best_model = True
        self.val_loss_min = val_loss
        state_dict = model.state_dict()
        for i in state_dict.keys():
            state_dict[i] = state_dict[i].to('cpu')
        #print(state_dict)
        torch.save(state_dict, path + '/' + 'model_' + str(tid))
        
        loss_log = torch.load(log_path)
        loss_log[tid] = self.val_loss_min
        torch.save(loss_log, log_path)
        print("common line69: model saved with val loss {}".format(self.val_loss_min))

    def __call__(self, val_loss, model, path, tid, log_path):
        # type: (nn.MSELoss, BaselineFCModel, str, int, str) -> None
        """
        Desc:
            __call__
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, tid, log_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.best_model = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_hidden = True
            self.save_checkpoint(val_loss, model, path, tid, log_path)
            self.counter = 0


class Experiment(object):
    """
    Desc:
        The experiment to train, validate and test a model
    """
    def __init__(self, args):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            args: the arguments to initialize the experimental environment
        """
        self.args = args

    def get_args(self):
        # type: () -> dict
        """
        Desc:
            Get the arguments
        Returns:
            A dict
        """
        return self.args

    def get_data(self, flag):
        # type: (str) -> (WindTurbineData, DataLoader)
        """
        Desc:
            get_data
        Args:
            flag: train or val
        Returns:
            A dataset and a dataloader
        """
        data_set = BaseWPFDataset(
            data_path=self.args["data_path"],
            filename=self.args["filename"],
            mean_path=self.args["path_to_mean"],
            std_path=self.args["path_to_std"],
            flag=flag,
            use_new_data=self.args["use_new_data"],
            size=[self.args["input_len"], self.args["output_len"]],
            step_size=self.args["step_size"],
            task=self.args["task"],
            target=self.args["target"],
            start_col=self.args["start_col"],
            columns=self.args["columns"],
            scale_cols=self.args["scale_cols"],
            turbine_id=self.args["turbine_id"],
            day_len=self.args["day_len"],
            prev_train_days=self.args["train_days"],
            new_train_days=self.args["actual_train_days"],
            prev_val_days=self.args["val_days"],
            new_val_days=self.args["actual_val_days"],
            prev_total_days=self.args["total_days"],
            new_total_days=self.args["actual_total_days"]
        )
        data_loader = DataLoader(
            data_set,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            drop_last=True
        )
        return data_set, data_loader

    def load_train_data(self):
        # type: () -> WindTurbineData
        """
        Desc:
            Load train data to get the scaler for testing
        Returns:
            The train set
        """
    
        train_data = BaseWPFDataset(
            data_path=self.args["data_path"],
            filename=self.args["filename"],
            mean_path=self.args["path_to_mean"],
            std_path=self.args["path_to_std"],
            flag='train',
            use_new_data=self.args["use_new_data"],
            size=[self.args["input_len"], self.args["output_len"]],
            step_size=self.args["step_size"],
            task=self.args["task"],
            target=self.args["target"],
            start_col=self.args["start_col"],
            columns=self.args["columns"],
            scale_cols=self.args["scale_cols"],
            day_len=self.args["day_len"],
            prev_train_days=self.args["train_days"],
            new_train_days=self.args["actual_train_days"],
            prev_val_days=self.args["val_days"],
            new_val_days=self.args["actual_val_days"],
            prev_total_days=self.args["total_days"],
            new_total_days=self.args["actual_total_days"],
            is_test=True
        )
        return train_data

    def get_optimizer(self, model):
        # type: (BaselineFCModel) -> torch.optimizer.Adam
        """
        Desc:
            Get the optimizer
        Returns:
            An optimizer
        """
        model_optim = torch.optim.Adam(params=model.parameters(),
                                       lr=self.args["lr"],
                                       weight_decay = 0.001)
                                            # grad_clip=clip)
                                            
        return model_optim

    @staticmethod
    def get_criterion():
        # type: () -> nn.MSELoss
        """
        Desc:
            Use the mse loss as the criterion
        Returns:
            MSE loss
        """
        criterion = nn.MSELoss(reduction='mean')
        return criterion

    def process_one_batch(self, model, batch_x, batch_y):
        # type: (BaselineFCModel, torch.tensor, torch.tensor) -> (torch.tensor, torch.tensor)
        """
        Desc:
            Process a batch
        Args:
            model:
            batch_x:
            batch_y:
        Returns:
            prediction and ground truth
        """
        device = self.args["device"]
        # batch_x = batch_x.reshape(batch_x.shape[0], -1)
        batch_x = batch_x.to(dtype=torch.float32)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(dtype=torch.float32)
        batch_y = batch_y.to(device)
        #
        sample = model(batch_x)
        #
        f_dim = -1 if self.args["task"] == 'MS' else 0
        #
        batch_y = batch_y[:, -self.args["output_len"]:, f_dim:].to(dtype=torch.float32)
        sample = sample.unsqueeze(dim=2).to(dtype=torch.float32)

        return sample, batch_y

    @staticmethod
    def get_test_x(args):
        # type: (dict) -> TestData
        """
        Desc:
            Obtain the input sequence for testing
        Args:
            args:
        Returns:
            Normalized input sequences and training data
        """
        test_x = TestData(path_to_data=args["path_to_test_x"], start_col=1, farm_capacity=args["capacity"])
        return test_x

    def inference_one_sample(self, model, sample_x):
        # type: (BaselineFCModel, torch.tensor) -> torch.tensor
        """
        Desc:
            Inference one sample
        Args:
            model:
            sample_x:
        Returns:
            Predicted sequence with sample_x as input
        """
        device = self.args["device"]
        # print("common line270 sample_x is: \n", sample_x[:5, :])
        x = sample_x.to(device)
        x = x.type(torch.float32)
       
        prediction = model(x)
        # clip
        # prediction[prediction <= 0] = 0
        # prediction[prediction >= 1500] = 1500
        # print("common line280 \n", prediction)
        prediction = prediction.reshape(self.args["output_len"], -1)
    
        f_dim = -1 if self.args["task"] == 'MS' else 0
        return prediction[..., :, f_dim:].type(torch.float32)


class Logger(object):
    def __init__(self, fileN="Default.log"):
        if os.path.exists(fileN):
            os.remove(fileN)
        self.terminal = sys.stdout
        self.log = open(fileN, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        pass