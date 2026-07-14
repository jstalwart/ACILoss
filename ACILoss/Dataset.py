from torch.utils.data import Dataset, Subset, DataLoader
from scipy.interpolate import CubicSpline
from einops import reduce
import torch.nn as nn
from .utils import *
from pywt import cwt 
import pandas as pd
import numpy as np
import scipy as sp
import torch

class MainDataset(Dataset):
    '''
    Parameters:
        - path (str): the path for the dataset file. 
        - endogenous (str): the variable to predict.
        - tau (int): the amount of endogenous data accounted.
        - H (int): the prediction horizon
        - partition (list of floats): the porcentage data splits (train, test, val). It must add up to 1.
        - dataset (pd.Series): the time series. 
        - splits (list of floats): the cummulative list of partition.
        - mode (str): the data partition name. Only 'train', 'test' and 'val' are accepted.
        - data (pd.Series): the data partition converted to rolling windows.
    '''
    def __init__(self,
                 path: str,
                 endogenous: str,
                 window_size: int,
                 horizon: int,
                 splits: list = [.8, .1, .1],
                 mode="train",
                 **kwargs):
        '''
        Initializes the dataset.
        Args:
            - path (str): the path for the dataset file. 
            - endogenous (str): the variable to predict.
            - window_size (int): the amount of endogenous data accounted.
            - horizon (int): the prediction horizon
            - splits (list of floats): the porcentage data splits (train, test, val). It must add up to 1.
            - mode (str): the data partition. Only 'train', 'test' and 'val' are accepted.
        '''
        super().__init__()
        assert np.sum(splits) == 1, "Splits summatory must add up to 1."
        self.path = path
        self.endogenous = endogenous
        self.tau = window_size
        self.H = horizon
        self.partition = splits

        self.dataset = pd.read_csv(path)[endogenous]
        self.splits = np.trunc(np.cumsum(splits)*len(self.dataset)).astype(int)
        self.select_split(mode)

    def __len__(self):
        return len(self.data)
    
    def _transform_to_sliding_window(self, series:pd.Series):
        return np.lib.stride_tricks.sliding_window_view(series, self.tau+self.H)
    
    def select_split(self, mode:str):
        '''
        Selects the split to use. 
        Args:
            - mode (str): the data partition. Only 'train', 'test' and 'val' are accepted.
        '''
        self.mode = mode.lower()
        if self.mode == "train":
            train = np.array(self.dataset[:self.splits[0]])
            self.data = self._transform_to_sliding_window(train)
        elif self.mode == "test":
            test = np.array(self.dataset[self.splits[0]:self.splits[1]])
            self.data = self._transform_to_sliding_window(test)
        elif self.mode == "val":
            val = np.array(self.dataset[self.splits[1]:])
            self.data = self._transform_to_sliding_window(val)
        else:
            raise ValueError(f"Mode {mode} is not well defined. Only 'train', 'test' and 'val' are accepted.")
        
    def copy(self, mode="train"):
        '''
        Returns a copy of the data. 
        Args:
            - mode (str): the data partition. Only 'train', 'test' and 'val' are accepted.
        Returns:
            - data (MainDataset): the copy of the data.
        '''
        return MainDataset(path=self.path, 
                           endogenous=self.endogenous,
                           window_size=self.tau,
                           horizon = self.H,
                           splits = self.partition,
                           mode = mode)

    def __norm__(self, x:np.array, y:np.array, eps: float = 1e-8)->tuple:
        '''
        Normalizes the data by instances.

        Args:
            - x (np.array): the train data to be normalized.
            - y (np.array): the target data to normalize. 

        Returns:
            - x (np.array): the train data normalized.
            - y (np.array): the target data normalized according to the train.
            - median (float): the median value from x.
            - std (float): the standard deviation from x.
        '''
        median = np.median(x)
        std = np.std(x)
        xi = (x-median)/(std+eps)
        yi = (y-median)/(std+eps)

        return xi, yi, median, std
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        x, y, median, std = self.__norm__(seq[:self.tau], seq[self.tau:])
        return {"idx": idx,
                "x": torch.from_numpy(x).float(),
                "y": torch.from_numpy(y).float(),
                "median": torch.Tensor([median]),
                "std": torch.Tensor([std])}
    

class TSDataset(Dataset):
    '''
    Parameters:
        - path (str): the path for the dataset file. 
        - endogenous (str): the variable to predict.
        - exogenous (list of str): other predicting variables.
        - MW (str): the mother wavelet emplyed.
        - tau (int): the amount of endogenous data accounted.
        - H (int): the prediction horizon
        - partition (list of floats): the porcentage data splits (train, test, val). It must add up to 1.
        - dataset (pd.Series): the time series. 
        - splits (list of floats): the cummulative list of partition.
        - mode (str): the data partition name. Only 'train', 'test' and 'val' are accepted.
        - data (dict variable:pd.Series): dictionary containinh the data partition converted to rolling windows.
    '''
    def __init__(self,
                 path: str,
                 endogenous: str,
                 exogenous: list,
                 window_size: int,
                 horizon: int,
                 splits: list = [.8, .1, .1],
                 mode="train",
                 **kwargs):
        '''
        Initializes the dataset.
        Args:
            - path (str): the path for the dataset file. 
            - endogenous (str): the variable to predict.
            - exogenous (list of str): other predicting variables. 
            - window_size (int): the amount of endogenous data accounted.
            - horizon (int): the prediction horizon
            - splits (list of floats): the porcentage data splits (train, test, val). It must add up to 1.
            - mode (str): the data partition. Only 'train', 'test' and 'val' are accepted.
        '''
        super().__init__()
        assert np.sum(splits) == 1, "Splits summatory must add up to 1."
        self.path = path
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.tau = window_size
        self.H = horizon
        self.partition = splits

        self.dataset = pd.read_csv(path)
        datetime = pd.to_datetime(self.dataset["date"])
        self.dataset['month'] = datetime.dt.month
        self.dataset['day'] = datetime.dt.day
        self.dataset['weekday'] = datetime.dt.weekday
        self.dataset['hour'] = datetime.dt.hour
        self.dataset['minute'] = datetime.dt.minute
        
        self.splits = np.trunc(np.cumsum(splits)*len(self.dataset)).astype(int)
        self.select_split(mode)
        self.splits = np.trunc(np.cumsum(splits)*len(self.dataset)).astype(int)
        self.select_split(mode)

    def __len__(self):
        return len(self.data) - self.H - self.tau
    
    def select_split(self, mode:str):
        '''
        Selects the split to use. 
        Args:
            - mode (str): the data partition. Only 'train', 'test' and 'val' are accepted.
        '''
        self.mode = mode.lower()
        if self.mode == "train":
            self.data = self.dataset[:self.splits[0]]
        elif self.mode == "test":
            self.data = self.dataset[self.splits[0]:self.splits[1]]
        elif self.mode == "val":
            self.data = self.dataset[self.splits[1]:]
        else:
            raise ValueError(f"Mode {mode} is not well defined. Only 'train', 'test' and 'val' are accepted.")
        
    def copy(self, mode="train"):
        '''
        Returns a copy of the data. 
        Args:
            - mode (str): the data partition. Only 'train', 'test' and 'val' are accepted.
        Returns:
            - data (MainDataset): the copy of the data.
        '''
        return TSDataset(path=self.path, 
                           endogenous=self.endogenous,
                           exogenous=self.exogenous,
                           window_size=self.tau,
                           horizon = self.H,
                           splits = self.partition,
                           mode = mode)

    def __norm__(self, x:np.array, y:np.array, eps: float = 1e-8)->tuple:
        '''
        Normalizes the data by instances.

        Args:
            - x (np.array): the train data to be normalized.
            - y (np.array): the target data to normalize. 

        Returns:
            - x (np.array): the train data normalized.
            - y (np.array): the target data normalized according to the train.
            - median (float): the median value from x.
            - std (float): the standard deviation from x.
        '''
        median = np.median(x)
        std = np.std(x)
        xi = (x-median)/(std+eps)
        if y is not None:
            yi = (y-median)/(std+eps)
            return xi, yi, median, std
        return xi, median, std
    
    def __getitem__(self, idx):
        '''
        Returns:
            - observation: dict {"idx": observatoin index,
                                 "x" :  the pseudo-images
                                 "y" : the }
        
        '''
        observation = {"x":[],
                       "z":[],
                       "median" : [],
                       "std" : []}
        #print(self.data.loc[self.endogenous, idx])
        aux = np.array(self.data[self.endogenous])
        x, y, median, std = self.__norm__(aux[idx:idx+self.tau], y = aux[idx+self.tau:idx+self.tau+self.H])
        observation["x"].append(x)
        observation["y"] = y
        observation["median"].append(median)
        observation["std"].append(std)

        aux = self.data.iloc[idx:idx+self.tau]

        for variable in self.exogenous:
            z, median, std = self.__norm__(np.array(aux[variable].astype(float)), y = None)
            observation["x"].append(z)
            observation["z"].append(z)
        observation["x"] = torch.from_numpy(np.array(observation["x"])).T
        observation["z"] = torch.from_numpy(np.array(observation["z"])).T
        mask = torch.cat([torch.ones(self.tau, len(self.exogenous)), torch.zeros(self.H, len(self.exogenous))], dim = 0).float()

        time_features = ['month', 'day', 'weekday', 'hour', 'minute']
        x_mark = aux[time_features].values # Shape: (window_size, 5)

        aux = self.data.iloc[idx:idx+self.tau+self.H]
        y_mark = aux[time_features].values

        return {"idx": idx,
                "x": observation["x"].float(),
                "y": torch.from_numpy(observation["y"]).float(),
                "z": torch.cat([observation["z"], torch.zeros(self.H, len(self.exogenous))], dim=0).float(),
                "t1": torch.from_numpy(x_mark).long(),
                "t2": torch.from_numpy(y_mark).long(),
                "mask": mask,
                "median": torch.Tensor(observation["median"]),
                "std": torch.Tensor(observation["std"])}
    

