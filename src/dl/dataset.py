import numpy as np
import torch

class DLDataset(torch.utils.data.Dataset):
    def __init__(self, df): #, config):
        self.df = df
        #self.config = config
        
        self.data = self._feature_engineering(df)
        
    def _feature_engineering(self, df):
        # Initial features
        High = df['High'].values
        Low = df['Low'].values
        Open = df['Open'].values
        Close = df['Close'].values
        Volume = df['Volume'].values
        
        # Percentage of Change
        
        # MFI
        MF = Volume * (High + Low + Close) / 3
        
        # Moving Everage
        
        # Bollinger Bands
        
        # Define features
        start_index = 20
        using_features = [High, Low, Open, Volume]
        x = np.array(using_features).T[start_index:-1]
        y = Close[start_index+1:].reshape([-1,1])
        data = np.hstack((x,y))
        
        return data
        
    def __getitem__(self, index):
        # x : column index 0 ~ -1, y : last column
        return torch.tensor(self.data[index])
    
    def __len__(self):
        return len(self.data)