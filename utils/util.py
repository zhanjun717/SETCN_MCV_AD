import torch
from torchvision import datasets, transforms
from sklearn import metrics
from sklearn.metrics import r2_score
import numpy as np

def data_generator(root, batch_size):
    train_set = datasets.MNIST(root=root, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    test_set = datasets.MNIST(root=root, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader


def metrics_calculate(y_orig , y_pred):
    mse = metrics.mean_squared_error(y_orig , y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_orig , y_pred)
    mape = np.mean(np.abs((y_orig - y_pred) / y_orig)) * 100
    smape = 2.0 * np.mean(np.abs(y_orig - y_pred) / (np.abs(y_pred) + np.abs(y_orig))) * 100
    res = y_orig - y_pred
    sd = np.std(res)
    print("MSE         |RMSE       |MAE        |MAPE       |SMAPE      |SD          ")
    print("{:<12.6}{:<12.6}{:<12.6}{:<12.6}{:<12.6}{:<12.6}".format(mse,rmse,mae,mape,smape,sd))

def cv_calculate(data):
    return np.std(data) / np.mean(data)

# MAPE和SMAPE需要自己实现
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100