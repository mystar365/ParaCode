# Standard library
import os
import random
import time
import warnings
from collections import namedtuple
from typing import Type, Union, List, Optional
import math
import pdb

# Third-party libraries
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.nn import (
    Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid,
    Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d,
    Sequential, Module, Parameter, Conv1d, MaxPool1d, AdaptiveAvgPool1d
)
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CosineAnnealingLR

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_squared_log_error
)

# Other utilities
from tqdm import tqdm
import torch.cuda
from torch.utils.data import Dataset, DataLoader

# Suppress warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mytool2: important tool 2

class PredictionDataset(Dataset):
    def __init__(self, features, spec_mean, spec_std):
        self.features = features.astype(np.float32)
        self.spec_mean = spec_mean
        self.spec_std = spec_std
        
    def __getitem__(self, index):
        x = (self.features[index] - self.spec_mean) / (self.spec_std + 1e-8)
        return torch.tensor(x, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)


    
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def mc_my_predict_candi(model, data_iter, lstd, lmean, mc_num):
    model.eval()
    enable_dropout(model) 
    with torch.no_grad():
        predictions = []
        targets = []
        uncertaintys = []        
        for x in data_iter:  
            x = x.to(device)
            batch_size = x.shape[0]  
            mid_outputs = np.zeros((mc_num, batch_size), dtype='float')
            for i in range(mc_num):
                output = model(x)  
                output = output.squeeze().cpu().numpy() * lstd + lmean  # (batch_size,)  
                mid_outputs[i] = output  
            mean_output = mid_outputs.mean(axis=0)  # (batch_size,)
            uncertainty = mid_outputs.std(axis=0)    # (batch_size,)
            predictions.extend(mean_output.tolist())
            uncertaintys.extend(uncertainty.tolist())
        
    predictions = np.array(predictions)
    uncertaintys = np.array(uncertaintys)
    return  predictions, uncertaintys


def mc_my_predict_candi_one(model, data_iter, lstd, lmean,mc_num):
    model.eval()
    enable_dropout(model) 
    with torch.no_grad():
        predictions = []
        targets = []
        uncertaintys = []        
        for x in data_iter:  
            x = x.to(device)
            batch_size = x.shape[0]  
            mid_outputs = np.zeros((mc_num, batch_size), dtype='float')
            for i in range(mc_num):
                output = model(x)  
                output = output.squeeze().cpu().numpy() * lstd + lmean  # (batch_size,)  
                mid_outputs[i] = output  
            mean_output = mid_outputs.mean(axis=0)  # (batch_size,)
            uncertainty = mid_outputs.std(axis=0)    # (batch_size,)
            predictions.extend(mean_output.tolist())
            uncertaintys.extend(uncertainty.tolist())   
    predictions = np.array(predictions)
    uncertaintys = np.array(uncertaintys)
    mid_outputs = np.array(mid_outputs)
    return  predictions, uncertaintys, mid_outputs


class SpectraDataset_prior(Dataset):
    def __init__(self, specdata, logg, logg_prob, logg_bias):
        self.specdata = specdata.reset_index(drop=True)
        self.spec_values = specdata.iloc[:, 0:3909].values.astype(np.float32)
        self.labels = specdata[logg].values.astype(np.float32)
        self.logg_prob = specdata[logg_prob].values
        self.logg_bias = specdata[logg_bias].values
        
    
        self.spec_mean = np.mean(self.spec_values, axis=0)
        self.spec_std = np.std(self.spec_values, axis=0)
        self.label_mean = np.mean(self.labels)
        self.label_std = np.std(self.labels)

    def __getitem__(self, index):
        specdata = (self.spec_values[index] - self.spec_mean) / (self.spec_std + 1e-8)
        label = (self.labels[index] - self.label_mean) / self.label_std
        logg_prob = self.logg_prob[index]
        logg_bias = self.logg_bias[index]
        
        return (
            torch.tensor(specdata, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(logg_prob, dtype=torch.float32),
            torch.tensor(logg_bias, dtype=torch.float32),     
        )

    def __len__(self):
        return len(self.specdata)

class SpectraDataset_weight(Dataset):
    def __init__(self, specdata, label, label_adaptive_weight, alldata):
        self.specdata = specdata.reset_index(drop=True)
        self.spec_values = specdata.iloc[:, 0:3909].values.astype(np.float32)
        self.labels = specdata[label].values.astype(np.float32)
        self.logg_adaptive_weight = specdata[label_adaptive_weight].values

        
        all_spec_values = alldata.iloc[:, 0:3909].values.astype(np.float32)
        all_labels = alldata[label].values.astype(np.float32)      
        self.spec_mean = np.mean(all_spec_values, axis=0)
        self.spec_std = np.std(all_spec_values, axis=0)
        self.label_mean = np.mean(all_labels)
        self.label_std = np.std(all_labels)

    def __getitem__(self, index):

        specdata = (self.spec_values[index] - self.spec_mean) / (self.spec_std)
        label = (self.labels[index] - self.label_mean) / self.label_std
        logg_adaptive_weight = self.logg_adaptive_weight[index]
        
        return (
            torch.tensor(specdata, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(logg_adaptive_weight, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.specdata)    
    

def evaluate_loss_stage2(model, data_iter, loss_fn_stage2, lstd, lmean):
    model.eval()
    with torch.no_grad():
        test_total_loss = 0
        predictions = []
        targets = []
        for x, y, w in data_iter:  
            x, y, w = x.to(device), y.to(device), w.to(device) 
            output = model(x)
            batch_loss = loss_fn_stage2(output.squeeze(), y, w)  
            test_total_loss += batch_loss.item()  
            predictions.extend(output.squeeze().cpu().numpy().tolist())
            targets.extend(y.cpu().tolist())  
    test_avg_loss = test_total_loss / len(data_iter.dataset)
    targets = np.array(targets)  
    predictions = np.array(predictions)  
    predictions = lstd * predictions + lmean
    targets = lstd * targets + lmean
    mae_test = mean_absolute_error(targets, predictions)  
    rmse_test = np.sqrt(mean_squared_error(targets, predictions))  
    return test_avg_loss, mae_test, rmse_test, predictions, targets
    
def train_stage2(model, train_loader, optimizer, scheduler, loss_fn, epochs,  save_dir, task, valid_loader, lstd, lmean, Num):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    loss_history_train = []
    loss_history_val = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        # start, total_loss = time.time(), 0.0
        # progess_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Progress bar')
        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs.squeeze(), y, w)
            loss.backward()
            optimizer.step()
            #print("Raw loss:", loss.item())  
            total_loss += loss.item()
        val_avg_loss, mae_val, rmse_val, predictions, targets = evaluate_loss_stage2(model, valid_loader, loss_fn, lstd, lmean)
        train_avg_loss = total_loss / len(train_loader.dataset)
        
        #print("len(data_loader.dataset)", len(data_loader.dataset))  
        #print("Raw train_avg_loss:", train_avg_loss)   
        
        loss_history_train.append(train_avg_loss)
        loss_history_val.append(val_avg_loss)
        scheduler.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('【轮次信息】epoch:%2s  train_aveloss:%.2f  val_aveloss:%.2f  val_rmse:%.2f  val_mae:%.2f  elapsed_time: %.2fs'
      % (epoch + 1, train_avg_loss, val_avg_loss, rmse_val, mae_val, elapsed_time))
        if (epoch + 1) % Num == 0:  
            model_path = os.path.join(save_dir, f"{task}_epoch_{epoch + 1}.bin")
            torch.save(model.state_dict(), model_path)
    return loss_history_train, loss_history_val, predictions, targets

def mypredict2(model, data_iter, lstd, lmean):
    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        for x, y, w in data_iter: 
            x, y, w  =  x.to(device), y.to(device), w.to(device)
            output = model(x)
            predictions.extend(output.squeeze().cpu().numpy().tolist())  
            targets.extend(y.cpu().tolist()) 
    targets = np.array(targets)  
    predictions = np.array(predictions)  
    predictions = lstd * predictions + lmean
    targets = lstd * targets + lmean
    refer_mae_test = mean_absolute_error(targets, predictions)  
    refer_rmse_test = np.sqrt(mean_squared_error(targets, predictions)) 

    return  refer_mae_test,refer_rmse_test, predictions, targets


class PredictionDataset(Dataset):
    def __init__(self, features, spec_mean, spec_std):
        self.features = features.astype(np.float32)
        self.spec_mean = spec_mean
        self.spec_std = spec_std
        
    def __getitem__(self, index):
        x = (self.features[index] - self.spec_mean) / (self.spec_std + 1e-8)
        return torch.tensor(x, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)



def train_stage2_early(model, train_loader, optimizer, scheduler, loss_fn, epochs, save_dir, task, valid_loader, lstd, lmean, patience=7):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    no_improve_epoch = 0
    loss_history_train = []
    loss_history_val = []
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs.squeeze(), y, w)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        
        val_avg_loss, mae_val, rmse_val, predictions, targets = evaluate_loss_stage2(model, valid_loader, loss_fn, lstd, lmean)
        
        train_avg_loss = total_loss / len(train_loader.dataset)
        loss_history_train.append(train_avg_loss)
        loss_history_val.append(val_avg_loss)
        
        
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_train_loss = train_avg_loss
            no_improve_epoch = 0
            best_model_state = model.state_dict().copy()
            
            
            model_path = os.path.join(save_dir, f"{task}_best_model.bin")
            torch.save(best_model_state, model_path)
            print(f"【最佳模型保存】验证损失改善至: {val_avg_loss:.4f}")
        else:
            no_improve_epoch += 1
        
        
        if no_improve_epoch >= patience:
            print(f"【早停触发】连续 {patience} 轮验证损失未改善")
            break
        
        scheduler.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print('【轮次信息】epoch:%2d  train_aveloss:%.4f  val_aveloss:%.4f  val_rmse:%.4f  val_mae:%.4f  elapsed_time: %.2fs'
              % (epoch + 1, train_avg_loss, val_avg_loss, rmse_val, mae_val, elapsed_time))
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"【训练完成】加载最佳模型 (验证损失: {best_val_loss:.4f})")
    
    return loss_history_train, loss_history_val, predictions, targets
    
    
        
if __name__ == "__main__":
    print('mytool')
    