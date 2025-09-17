# mytool3.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def create_dataset(data_file_path, f, rand, device='cpu'):
  
    mydata = pd.read_csv(data_file_path)
    print("Total data information-->", mydata.info(), type(mydata))  
    
   
    total_data = np.array(mydata.iloc[:, :3909])
    labels = np.array(mydata[f])
    
  
    mean = np.mean(total_data, axis=0)
    std = np.std(total_data, axis=0)
    total_data = (total_data - mean) / std
    
    
    lmean = np.mean(labels, axis=0)
    lstd = np.std(labels, axis=0)
    labels = (labels - lmean) / lstd
    
   
    x_train, x_temp, y_train, y_temp = train_test_split(
        total_data, labels, test_size=0.2, random_state= rand)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state= rand)
    
    print(f"Train samples: {len(x_train)}")
    print(f"Val samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    
   
    def prepare_data(x_data):
        x_data = x_data.reshape(len(x_data), 3909)
        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.transpose(x_data, (0, 2, 1))
        return torch.from_numpy(x_data).float().to(device)
    
    x_train_tensor = prepare_data(x_train)
    x_val_tensor = prepare_data(x_val)
    x_test_tensor = prepare_data(x_test)
    
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)
    
  
    dataset_train = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    dataset_val = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
    dataset_test = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
    
    return dataset_train, dataset_val, dataset_test, x_train.shape[1], lmean, lstd


def create_dataset_weight(data_file_path, f, rand, w, device='cpu'):

    mydata = pd.read_csv(data_file_path)
    
   
    total_data = np.array(mydata.iloc[:, :3909])  
    labels = np.array(mydata[f])                  
    weights = np.array(mydata[w])          

    mean = np.mean(total_data, axis=0)
    std = np.std(total_data, axis=0)
    total_data = (total_data - mean) / (std + 1e-8)
    
    lmean = np.mean(labels)
    lstd = np.std(labels)
    labels = (labels - lmean) / lstd

   
    x_train, x_temp, y_train, y_temp, w_train, w_temp = train_test_split(
        total_data, labels, weights, test_size=0.2, random_state=rand)
    x_val, x_test, y_val, y_test, w_val, w_test = train_test_split(
        x_temp, y_temp, w_temp, test_size=0.5, random_state=rand)

    print(f"Train samples: {len(x_train)}")
    print(f"Val samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")

   
    def prepare_data(x_data):
        x_data = x_data.reshape(len(x_data), 3909)
        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.transpose(x_data, (0, 2, 1))
        return torch.from_numpy(x_data).float().to(device)
    
    x_train_tensor = prepare_data(x_train)
    x_val_tensor = prepare_data(x_val)
    x_test_tensor = prepare_data(x_test)

    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)

    w_train_tensor = torch.from_numpy(w_train).float().to(device)
    w_val_tensor = torch.from_numpy(w_val).float().to(device)
    w_test_tensor = torch.from_numpy(w_test).float().to(device)

   
    dataset_train = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor, w_train_tensor)
    dataset_val = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor, w_val_tensor)
    dataset_test = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor, w_test_tensor)

    return dataset_train, dataset_val, dataset_test, x_train.shape[1], lmean, lstd
