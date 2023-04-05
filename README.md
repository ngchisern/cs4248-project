# cs4248-project: Labeled Unreliable News
> Chi Sern, Kian En, En En, Ethan Wong, Chester Soh, Jayson Lam

This repository contains the source code and model predictions for our project. The goal of the project is to classify news articles into one of "Satire", "Hoax", "Propaganda", and "Reliable News".

## Data

The dataset used in this project consists of fulltrain.csv and balancedtest.csv, which can be obtained from [here](https://drive.google.com/drive/folders/1tZTCj5YhmOAoxiv078LmmwYnCMcBQaaq?usp=sharing).

## Models

The trained models used in this project are also too large to be included in this repository. The trained XLNet, RNN and LSTM models can be obtained from [here](https://drive.google.com/drive/folders/1TBGME3lL7DEv7XQwdG-c8Biq8wxZOGmV?usp=share_link).

## Source Code

The source code for this project is organized as follows:

- `model_training`: This folder contains Python scripts or Jupyter Notebooks for training the models end to end.
- `feature engineering`: This file is responsible for analyzing the training data and discovering useful features.
- `ensemble`: This file ensembles the XLNet, RNN, and LSTM models for generating predictions on balancedtest.csv. It will also create files for each individual model as well as the ensemble models. These prediction files can also be found in the `predictions` folder.
- `explainer`: This file utilizes [SHAP](https://github.com/slundberg/shap) techniques to generate (local) explanations for our models.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
