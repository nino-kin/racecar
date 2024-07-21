#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
This script contains functions and classes to load and preprocess data, define a neural network model, train the model, save the model, and test the model using PyTorch.

Modules imported:
- os
- time
- sys
- datetime
- pandas as pd
- torch
- torch.nn as nn
- torch.nn.functional as F
- torch.utils.data as DataLoader
- matplotlib.pyplot as plt
- config

Functions:
- load_data(): Loads and preprocesses the data from CSV files.
- normalize_ultrasonics(): Normalizes ultrasonic sensor data.
- denormalize_ultrasonics(): Denormalizes ultrasonic sensor data.
- normalize_motor(): Normalizes motor data.
- denormalize_motor(): Denormalizes motor data.
- steering_shifter_to_01(): Converts steering values from -1~1 to 0~1.
- steering_shifter_to_m11(): Converts steering values from 0~1 to -1~1.
- save_model(): Saves the model state.
- load_model(): Loads a saved model state.
- train_model(): Trains the model.
- test_model(): Tests the model by making predictions on a sample of data.
- main(): The main function to run the entire process.

Classes:
- CustomDataset: A custom dataset class for PyTorch.
- NeuralNetwork: A neural network model class for PyTorch.
"""

import os
import time
import sys
import datetime
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import config

def load_data():
    """
    Loads and preprocesses data from CSV files.

    Returns:
    - x_tensor (torch.Tensor): Input data tensor.
    - y_tensor (torch.Tensor): Target data tensor.
    - csv_file (str): Name of the CSV file used.
    """
    folder = "records"
    csv_files = [file for file in os.listdir(folder) if file.endswith(".csv")]
    print(csv_files)

    if len(csv_files) > 1:
        answer = input("\n複数のcsvファイルがあります。ファイルを結合しますか? (y)").strip().lower()
        if answer == "y":
            dataframes = []
            dataframe_columns = []
            for csv_file in csv_files:
                csv_path = os.path.join(folder, csv_file)
                df = pd.read_csv(csv_path)
                dataframes.append(df)
                dataframe_columns.append(df.columns)
                if len(dataframe_columns) > 1 and not all(dataframe_columns[0] == dataframe_columns[1]):
                    print(csv_path, "の列が他のファイルと異なり結合できません。確認してください。")
                    sys.exit()
            df = pd.concat(dataframes)
        else:
            csv_file = input("ファイル名を入力してください, Enterで最後を選択: ").strip()
            if csv_file == "":
                csv_file = csv_files[-1]
                print("\n最新のファイルを選択:", csv_file)
                time.sleep(0.5)
            csv_path = os.path.join(folder, csv_file)
            df = pd.read_csv(csv_path)
    else:
        csv_file = csv_files[0]
        csv_path = os.path.join(folder, csv_file)
        df = pd.read_csv(csv_path)

    print("\n入力データの確認:\n", df.head(3), "\nデータサイズ", df.shape)

    x = df.iloc[:, 3:]
    x_tensor = torch.tensor(x.values, dtype=torch.float32)
    x_tensor = normalize_ultrasonics(x_tensor)

    if config.model_type == "categorical":
        df['Str'] = pd.cut(df['Str'], bins=config.bins_Str, labels=False)
        y = df.iloc[:, 1:3]
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        print("\n学習データの確認:\n教師データ(ラベル:Str, Thr[学習なし]):", y_tensor[0, :], "\n入力データ(正規化センサ値):", x_tensor[0, :])
        print("学習データサイズ:", "y:", y_tensor.shape, "x:", x_tensor.shape, "\n")
    else:
        y = df.iloc[:, 1:3]
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        y_tensor = normalize_motor(y_tensor)
        y_tensor[:, 0] = steering_shifter_to_01(y_tensor[:, 0])
        print("\n学習データの確認:\n教師データ(正規化操作値+0.5: Str, Thr):", y_tensor[0, :], "\n入力データ(正規化センサ値):", x_tensor[0, :])
        print("学習データサイズ:", "y:", y_tensor.shape, "x:", x_tensor.shape, "\n")

    return x_tensor, y_tensor, csv_file

def normalize_ultrasonics(x_tensor, scale=2000):
    """
    Normalizes ultrasonic sensor data.

    Args:
    - x_tensor (torch.Tensor): Tensor of ultrasonic data.
    - scale (int): Scale factor for normalization.

    Returns:
    - torch.Tensor: Normalized ultrasonic data.
    """
    return x_tensor / scale

def denormalize_ultrasonics(x_tensor, scale=2000):
    """
    Denormalizes ultrasonic sensor data.

    Args:
    - x_tensor (torch.Tensor): Tensor of normalized ultrasonic data.
    - scale (int): Scale factor for denormalization.

    Returns:
    - torch.Tensor: Denormalized ultrasonic data.
    """
    return x_tensor * scale

def normalize_motor(y_tensor, scale=100):
    """
    Normalizes motor data.

    Args:
    - y_tensor (torch.Tensor): Tensor of motor data.
    - scale (int): Scale factor for normalization.

    Returns:
    - torch.Tensor: Normalized motor data.
    """
    return y_tensor / scale

def denormalize_motor(y_tensor, scale=100):
    """
    Denormalizes motor data.

    Args:
    - y_tensor (torch.Tensor): Tensor of normalized motor data.
    - scale (int): Scale factor for denormalization.

    Returns:
    - torch.Tensor: Denormalized motor data.
    """
    return y_tensor * scale

def steering_shifter_to_01(y_tensor):
    """
    Converts steering values from -1~1 to 0~1.

    Args:
    - y_tensor (torch.Tensor): Tensor of steering data.

    Returns:
    - torch.Tensor: Converted steering data.
    """
    return (y_tensor + 1) / 2

def steering_shifter_to_m11(y_tensor):
    """
    Converts steering values from 0~1 to -1~1.

    Args:
    - y_tensor (torch.Tensor): Tensor of steering data.

    Returns:
    - torch.Tensor: Converted steering data.
    """
    return (y_tensor - 0.5) * 2

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for PyTorch.

    Args:
    - x_tensor (torch.Tensor): Input data tensor.
    - y_tensor (torch.Tensor): Target data tensor.
    """
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    """
    Neural network model class for PyTorch.

    Args:
    - input_dim (int): Dimension of the input data.
    - output_dim (int): Dimension of the output data.
    - hidden_dim (int): Dimension of the hidden layers.
    - num_hidden_layers (int): Number of hidden layers.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        if config.model_type == "categorical":
            layers.append(nn.Linear(hidden_dim, config.num_categories))
        else:
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x = self.layers(x)
        if config.model_type == "categorical":
            x = F.log_softmax(x, dim=1)
        return x

    def predict(self, model, x_tensor):
        """
        Predict using the trained model.

        Args:
        - model (nn.Module): Trained model.
        - x_tensor (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Predictions.
        """
        model.eval()
        with torch.no_grad():
            predictions = model(x_tensor)
            if config.model_type == "categorical":
                predictions = torch.argmax(predictions, dim=1)
                predictions[0] = config.categories_Str[predictions[0]] / 100
                predictions = torch.tensor([[predictions[0], config.categories_Thr[predictions[0]] / 100]])
            else:
                predictions[:, 0] = steering_shifter_to_m11(predictions[:, 0])
        predictions = torch.clamp(predictions, min=-1, max=1)
        return predictions

    def predict_label(self, model, x_tensor):
        """
        Predict labels using the trained model.

        Args:
        - model (nn.Module): Trained model.
        - x_tensor (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Predicted labels.
        """
        model.eval()
        with torch.no_grad():
            predictions = model(x_tensor)
            predictions = torch.argmax(predictions, dim=1)
        return predictions

def train_model(model, dataloader, criterion, optimizer, start_epoch=0, epochs=config.epochs):
    """
    Train the model.

    Args:
    - model (nn.Module): Model to be trained.
    - dataloader (DataLoader): DataLoader for the training data.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - start_epoch (int): Starting epoch.
    - epochs (int): Number of epochs to train.

    Returns:
    - int: Final epoch.
    """
    model.train()
    loss_history = []
    for epoch in range(start_epoch, start_epoch + epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if config.model_type == "categorical":
                targets = targets[:, 0]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        loss_history.append(loss.item())
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    print("トレーニングが完了しました。")

    plt.figure()
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_history_path = config.model_dir + '/' + 'loss_history.png'
    plt.savefig(loss_history_path)
    plt.close()
    print("Lossの履歴を保存しました: " + loss_history_path)
    return epoch + 1

def save_model(model, optimizer, folder, csv_file, epoch):
    """
    Save the model state.

    Args:
    - model (nn.Module): Trained model.
    - optimizer: Optimization algorithm.
    - folder (str): Folder to save the model.
    - csv_file (str): Name of the CSV file used.
    - epoch (int): Epoch number.

    Returns:
    - str: Path of the saved model.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    model_name = f'model_{date_str}_{csv_file}_epoch_{epoch}_{config.ultrasonics_list_join}.pth'
    model_path = os.path.join(folder, model_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
    print(f"モデルを保存しました: {model_name}")
    return model_path

def load_model(model, model_path=None, optimizer=None, folder='.'):
    """
    Load a saved model state.

    Args:
    - model (nn.Module): Model instance.
    - optimizer: Optimization algorithm instance (optional).
    - folder (str): Folder where the model is saved.

    Returns:
    - int: Epoch number of the loaded model (0 if failed).
    """
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("オプティマイザの状態も読み込みました。")
        print(f"モデルを読み込みました: {model_path}")
        return checkpoint.get('epoch', 0)
    else:
        model_files = [file for file in os.listdir(folder) if file.startswith('model_')]
        if model_files:
            print("利用可能なモデル:")
            print(model_files)
            model_name = input("読み込むモデル名を入力してください.\n[WARN] 過去にモデル構造を変更している場合は読み込めませんので、config.pyを編集してください。\n: ")
            model_path = os.path.join(folder, model_name)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("オプティマイザの状態も読み込みました。")
            print(f"モデルを読み込みました: {model_name}")
            return checkpoint.get('epoch', 0)
        else:
            print("利用可能なモデルが見つかりませんでした。")
            return 0

def test_model(model, model_path, dataset, sample_num=5):
    """
    Test the model by making predictions on a sample of data.

    Args:
    - model (nn.Module): Trained model.
    - model_path (str): Path to the saved model.
    - dataset (CustomDataset): Dataset for testing.
    - sample_num (int): Number of samples to test.
    """
    model_dir = "models"
    print("\n保存したモデルをロードします。")
    load_model(model, model_path, None, model_dir)
    print(model)

    print("\n推論の実行例です。\nランダムに", sample_num, "コのデータを取り出して予測します。")
    testloader = DataLoader(dataset, batch_size=1, shuffle=True)
    tmp = iter(testloader)
    x = torch.tensor([])
    y = torch.tensor([])
    yh = torch.tensor([])
    for _ in range(sample_num):
        x1, y1 = next(tmp)
        x = torch.cat([x, x1])
        y = torch.cat([y, y1])
        if config.model_type == "linear":
            yh1 = model.predict(model, x1)
            yh = torch.cat([yh, yh1])
        elif config.model_type == "categorical":
            yh1 = model.predict_label(model, x1)
            yh = torch.cat([yh, torch.tensor([yh1, config.categories_Str[yh1]]).unsqueeze(0)])

    print("\n入力データ:")
    print(x)
    print("\n正解データ:")
    print(y)
    print("\n予測結果:")
    print(yh)
    if config.model_type == "categorical":
        print("\n正解率_Str: ", int(torch.sum(y[:, 0] == yh[:, 0]).item() / sample_num * 100), "%")
        print("confusion matrix_Str:\n", pd.crosstab(y[:, 0], yh[:, 0], rownames=['True'], colnames=['Predicted'], margins=True))
        print("\n正解率_Thr: ", int(torch.sum(y[:, 1] == yh[:, 1]).item() / sample_num * 100), "%")

    print("\n使用したモデル名:", os.path.split(model_path)[-1])

def main():
    """
    Main function to load data, create and train the model, save the model, and test the model.
    """
    x_tensor, y_tensor, csv_file = load_data()

    dataset = CustomDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    input_dim = x_tensor.shape[1]
    output_dim = y_tensor.shape[1]
    model = NeuralNetwork(input_dim, output_dim, config.hidden_dim, config.num_hidden_layers)
    print("モデル構造: ", model)

    if config.model_type == "categorical":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    continue_training = input("続きから学習を再開しますか？ (y): ").strip().lower() == 'y'
    start_epoch = 0

    if continue_training:
        start_epoch = load_model(model, None, optimizer, 'models')
    else:
        start_epoch = 0
    try:
        epochs = int(input(f"学習するエポック数を入力してください.(デフォルト:{config.epochs}): ").strip())
    except ValueError:
        epochs = config.epochs

    epoch = train_model(model, dataloader, criterion, optimizer, start_epoch=start_epoch, epochs=epochs)

    model_path = save_model(model, optimizer, 'models', csv_file, epoch)

    test_model(model, model_path, dataset)

if __name__ == "__main__":
    main()
