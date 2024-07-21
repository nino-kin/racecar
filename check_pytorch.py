#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from train_pytorch import load_data, CustomDataset, NeuralNetwork, test_model
import os
import sys
import time

import config

def main():
    """
    Main function to load data, create a model, and test it.

    This function performs the following steps:
    1. Load the data
    2. Create a dataset
    3. Create a neural network model
    4. Test the model (with an option to choose a different model)
    """
    # Load data
    x_tensor, y_tensor, csv_file = load_data()

    # Create dataset
    dataset = CustomDataset(x_tensor, y_tensor)

    # Create model
    input_dim = x_tensor.shape[1]
    output_dim = y_tensor.shape[1]
    model = NeuralNetwork(input_dim, output_dim, config.hidden_dim, config.num_hidden_layers)

    # Test model
    print("Starting model test...")
    print("Model path: ", config.model_path)
    answer = input("\nDo you want to test a different model? (y)")
    if answer == "y":
        folder = "models"
        models = [m for m in os.listdir(folder) if m.endswith(".pth")]
        print(models)
        if len(models) > 1:
            answer = input("\nEnter the model name to test, or press Enter to select the latest: ")
            if answer == "":
                answer = models[-1]
                print("\nSelected the latest file:", answer)
                time.sleep(0.5)
            config.model_path = os.path.join(folder, answer)
        else:
            print("No models found.")
            sys.exit()
    test_model(model, config.model_path, dataset, x_tensor.shape[0])

if __name__ == "__main__":
    main()
