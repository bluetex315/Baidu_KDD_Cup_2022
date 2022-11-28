import os
import sys
import time
import numpy as np
from typing import Callable
import torch
import random
from torch.utils.data import DataLoader
from model_old import BaselineFCModel
from model_rnn import BaselineRNNModel
# from model_gru import BaselineGRUModel
from model_lstm import BaselineLSTMModel
from common import EarlyStopping
from common import adjust_learning_rate
from common import Experiment
from common import Logger
from prepare import prep_env

def val(experiment, model, data_loader, criterion):
    # type: (Experiment, BaselineGRUModel, DataLoader, Callable) -> np.array
    """
    Desc:
        Validation function
    Args:
        experiment:
        model:
        data_loader:
        criterion:
    Returns:
        The validation loss
    """
    validation_loss = []
    model.eval()
    for i, (batch_x, batch_y) in enumerate(data_loader):
        # if i == 0:
        #     print("train line35", batch_x)
        sample, true = experiment.process_one_batch(model, batch_x, batch_y)
        loss = criterion(sample, true)
        validation_loss.append(loss.item())
    validation_loss = np.average(validation_loss)
    return validation_loss


def train_and_val(experiment, model, model_folder, is_debug=False, log_path=None):
    # type: (Experiment, BaselineGRUModel, str, bool, str) -> None
    """
    Desc:
        Training and validation
    Args:
        experiment:
        model:
        model_folder: folder name of the model
        is_debug:
    Returns:
        None
    """
    args = experiment.get_args()
    device = args["device"]

    train_data, train_loader = experiment.get_data(flag='train')
    print("\n train line60 ************** train loading finished**************** \n")
    val_data, val_loader = experiment.get_data(flag='val')

    path_to_model = os.path.join(args["checkpoints"], model_folder)
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)

    early_stopping = EarlyStopping(patience=args["patience"], verbose=True)
    optimizer = experiment.get_optimizer(model)
    criterion = Experiment.get_criterion()

    epoch_start_time = time.time()
    min_loss = np.inf
    for epoch in range(args["train_epochs"]):
        iter_count = 0
        train_loss = []
        model = model.to(device)
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            # if i == 0 & epoch == 0:
            #     print("train line80 \n", batch_x)
            iter_count += 1
            sample, truth = experiment.process_one_batch(model, batch_x, batch_y)
            loss = criterion(sample, truth)
            train_loss.append(loss.item())
            # Adam
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss = val(experiment, model, val_loader, criterion)
        if is_debug:
            train_loss = np.average(train_loss)
            epoch_end_time = time.time()
            print("\nEpoch: {}, \nTrain Loss: {}, \nValidation Loss: {}".format(epoch, train_loss, val_loss))
            print("Elapsed time for epoch-{}: {}".format(epoch, epoch_end_time - epoch_start_time))
            epoch_start_time = epoch_end_time

        if val_loss < min_loss:

            min_loss = val_loss

        # Early Stopping if needed
        early_stopping(val_loss, model, path_to_model, args["turbine_id"], log_path)
        if early_stopping.early_stop:
            print("Early stopped! ")
            break
    print("\ntrain line101: min loss for the epoch {} is {}".format(epoch, min_loss))
    

if __name__ == "__main__":
    cur_time = time.localtime()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", cur_time)
    sys.stdout = Logger('./train_log_{}.txt'.format(str(cur_time)))
    loss_log = {}
    log_path = './train_log_{}.pt'.format(str(cur_time))
    torch.save(loss_log,log_path)

    fix_seed = 3407

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    settings = prep_env()
    # Set up the initial environment
    # Current settings for the model
    cur_setup = '{}_t{}_i{}_o{}_newdata{}_time{}'.format(
        settings["filename"], settings["task"], settings["input_len"], settings["output_len"],
        settings["use_new_data"], str(cur_time)
    )
    
    start_train_time = time.time()
    end_train_time = start_train_time
    start_time = start_train_time
    for tid in range(settings["capacity"]):
        settings["turbine_id"] = tid
        exp = Experiment(settings)
        print('\n>>>>>>>>> Training Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(tid))
        baseline_model = BaselineRNNModel(settings)
        train_and_val(exp, model=baseline_model, model_folder=cur_setup, is_debug=settings["is_debug"], log_path=log_path)
        torch.cuda.empty_cache()
        if settings["is_debug"]:
            end_time = time.time()
            print("\nTraining the {}-th turbine in {} secs".format(tid, end_time - start_time))
            start_time = end_time
            end_train_time = end_time
    if settings["is_debug"]:
        print("\nTotal time in training {} turbines is "
              "{} secs".format(settings["capacity"], end_train_time - start_train_time))
