# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
"""
import torch

def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "path_to_phase1_x": "/home/chenlihui/KDD/kdd/test_x/0001in.csv",
        "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
        "data_path": "../data",
        "path_to_mean": "mean_folder",
        "path_to_std": "std_folder",
        "filename": "wtbdata_245days.csv",
        "actual_filename": "wtbdata_259days.csv",
        "use_new_data": False,
        "task": "MS",
        "target": "Patv",
        "checkpoints": "checkpoints",
        "turbine_id": 0,
        "input_len": 36,
        "step_size": 1, # we take 1 from every n observations, so the actual input length is input_len / step_size
        "output_len": 288,
        "columns": ["Day", "Tmstamp", "Wspd", "Wdir", "Etmp", "Patv"],
        "column_pos": [1, 2, 3, 4, 5, 12], # for `predict.py` evaluation use, since column names cannot be indexed in np.ndarray
        "start_col": 1,
        "scale_cols": ["Wspd", "Wdir", "Etmp", "Patv"],
        "in_var": 6,
        "out_var": 1,
        "day_len": 144,
        "train_days": 214,
        "actual_train_days": 228,
        "val_days": 31,
        "actual_val_days": 31,
        "total_days": 245,
        "actual_total_days": 259,
        "gru_hidden_size": 12,
        "gru_layers": 4,
        "dropout": 0.05,
        "num_workers": 0,
        "train_epochs": 32,
        "batch_size": 64,
        "patience": 12,
        "lr": 0.25e-4,
        "lr_adjust": "type1",
        "device": "cpu",
        "capacity": 134,
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "pytorch",
        "is_debug": True
    }

    # Prepare the GPUs
    if torch.cuda.is_available():
        settings["device"] = 0
    else:
        settings["device"] = 'cpu'
    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
