# Spatial-Temporal Recurrent Neural Network for Wind Power Forecasting in Baidu KDD CUP 2022

This repository contains the Baidu KDD CUP2022 solution by Lihui Chen and Jiangyi Zhu.  

Our team name is **Zealen**, winning 32th place in 2490 teams. 

Baidu also offers a preliminary solution by a simple GRU model, baseline codes should refers to the [official baseline](https://github.com/PaddlePaddle/PGL/tree/main/examples/kddcup2022/wpf_baseline).

## Overview

Wind power is a rapidly growing source of clean energy. Accurate wind power forecasting is essential for grid stability and the security of supply. Therefore, organizers provide a wind power dataset containing historical data from 134 wind turbines and launch the [Baidu KDD Cup 2022](https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction) to examine the limitations of current methods for wind power forecasting. The average of RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) is used as the evaluation score. 

We adopt two spatial-temporal graph neural network models, i.e., AGCRN and MTGNN, as our basic models. We train AGCRN by 5-fold cross-validation and additionally train MTGNN directly on the training and validation sets. Finally, we ensemble the two models based on the loss values of the validation set as our final submission. Using our method, our team **BUAA\_BIGSCity** achieves -45.36026 on the test set.

## Train

First place the [data](https://aistudio.baidu.com/aistudio/competition/detail/152/0/datasets) in the `data/` directory. Then run the script `gen_graph.py` to generate the geographic distance graph. After this, you can train the models.

### MTGNN

You can run the following command to train the MTGNN model on a 214-day training sets and a 31-day validation sets.

```shell
python train.py
```

### Fusion

After training, we perform a weighted fusion of the prediction results of the 5 AGCRN models based on the reciprocals of valid losses. After obtaining the ensembled AGCRN model, we integrate the ensembled AGCRN model and MTGNN model again according to the ratio of **4:6** and obtain the final model prediction results. 

Using this method, we achieve **-45.36026** on the test set finally. The `submit.zip` here is our final submission file.
