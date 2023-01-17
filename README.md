# Spatial-Temporal Recurrent Neural Network for Wind Power Forecasting in Baidu KDD CUP 2022

This repository contains the Baidu KDD CUP2022 solution by Lihui Chen and Jiangyi Zhu.  

Our team name is **Zealen**, winning 32th place in 2490 teams. 

Baidu also offers a preliminary solution by a simple GRU model, baseline codes should refers to the [official baseline](https://github.com/PaddlePaddle/PGL/tree/main/examples/kddcup2022/wpf_baseline).

## Overview

Wind power is a rapidly growing source of clean energy. Accurate wind power forecasting is essential for grid stability and the security of supply. Therefore, organizers provide a wind power dataset containing historical data from 134 wind turbines and launch the [Baidu KDD Cup 2022](https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction) to examine the limitations of current methods for wind power forecasting. The average of RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) is used as the evaluation score. 

We adopted two recurrent neural network models, i.e., plain RNN and GRU, as our basic models. Those two models was trained separately by 5-fold cross-validation. Finally, we ensemble the two models based on the loss of the validation set as our final submission. Our team **Zealen** has achieved -46.13 on the final test set.

## Prepare

First place the [data](https://aistudio.baidu.com/aistudio/competition/detail/152/0/datasets) in the `data/` directory.

All the parameters, for instance \# of layers, \# of features used, are defined in `prepare.py` in dictionary. If you want to run the experiments on your own, please modify `prepare.py` accordingly.

## Train

You can run the following command to train the RNN or GRU model on a 214-day training set and a 31-day validation set.

```shell
python train.py
```

Currently in `train.py` script the model is specified to be plain RNN. You should modify the `prepare.py` in order to change the model to GRU.

### Fusion

After training, we perform a weighted fusion of the prediction results of the RNN and GRU model based on the loss in validation set. We additionally enumerated the best ratio of RNN and GRU predictions, then combined the two models' prediction according to that specific ratio. 

Using this method, we achieve **-46.13** on the test set finally.
