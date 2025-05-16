from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import torch
from statsmodels.tsa.seasonal import STL

import numpy as np
import os

dataset_path = "npformat"
os.environ["OMP_NUM_THREADS"] = "8"

cuda = True
if cuda == True and torch.cuda.is_available():
    device = torch.device("cuda")
    print("=== Using CUDA ===")
else:
    if cuda == True and not torch.cuda.is_available():
        print("=== CUDA is unavailable ===")
    device = torch.device("cpu")
    print("=== Using CPU ===")
    
def decompose(
    x, period: int = 7
):
    decomposed = STL(x, period=period).fit()
    trend = decomposed.trend.astype(np.float32)
    seasonal = decomposed.seasonal.astype(np.float32)
    residual = decomposed.resid.astype(np.float32)
    return torch.stack([torch.from_numpy(trend),
        torch.from_numpy(seasonal),
        torch.from_numpy(residual)], dim=0)

def read_sets(sets):
    array_list = []
    array_info = []
    for one_set in sets:
        datas = np.load("{}/{}.npz".format(dataset_path, one_set))
        array_names = datas.files
        for array_name in array_names:
            array_list.append(datas[array_name])
            
            if one_set[:4] == "ETTh":
                array_info.append(24)
            elif one_set[:4] == "ETTm":
                array_info.append(96)
            elif one_set[:4] == "exch":
                array_info.append(7)
            elif one_set[:4] == "weat":
                array_info.append(144)
            else:
                array_info.append(24)
            
    return array_list, array_info

class UnivDataset(Dataset):
    def __init__(self, train_sets, seq_len, pred_len, train_prop=0.7, valid_prop=0.1, phase="train", unsqueeze=False) -> None:
        super().__init__()
        
        self.data, _ = read_sets(train_sets)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.unsqueeze = unsqueeze
        
        new_array_list = []
        idx_map = []
        idx_start = 0
        
        for i in range(len(self.data)):
        # for i in range(1):
        #     i = len(self.data) - 1
            # i = 0
            
            scaler = StandardScaler()
            
            train_data = self.data[i][:int(train_prop * len(self.data[i]))].reshape(-1, 1)
            scaler.fit(train_data)
            if phase == "train":
                self.data[i] = scaler.transform(train_data)
            elif phase == "valid":
                valid_data = self.data[i][int(train_prop * len(self.data[i])) - seq_len: int((train_prop + valid_prop) * len(self.data[i]))]
                self.data[i] = scaler.transform(valid_data.reshape(-1, 1))
            elif phase == "test":
                test_data = self.data[i][int((train_prop + valid_prop) * len(self.data[i])) - seq_len:]
                self.data[i] = scaler.transform(test_data.reshape(-1, 1))
            else:
                raise NotImplementedError("Wrong phase.")
            
            self.data[i] = self.data[i].flatten()
            
            array_len = len(self.data[i])
            
            # to small dataset
            if array_len < pred_len + seq_len:
                continue
            
            # fixing missing value
            nan_indices = np.where(np.isnan(self.data[i]))[0]
            nan_len = len(nan_indices)
            
            if nan_len == 0:
                new_array_list.append(self.data[i])
                idx_map.append(np.arange(idx_start, idx_start + len(self.data[i]) - seq_len - pred_len + 1))
                idx_start += len(self.data[i])
                
            else:
                # filling one point missing value
                if nan_indices[0] > 0 and nan_indices[1] != nan_indices[0] + 1:
                    self.data[i][nan_indices[0]] = (self.data[i][nan_indices[0] - 1] + self.data[i][nan_indices[0] + 1]) / 2
                for j in range(1, nan_len - 1):
                    if nan_indices[j - 1] != nan_indices[j] - 1 and nan_indices[j + 1] != nan_indices[j] + 1:
                        self.data[i][nan_indices[j]] = (self.data[i][nan_indices[j] - 1] + self.data[i][nan_indices[j] + 1]) / 2
                if nan_indices[-1] < array_len - 1 and nan_indices[-1] != nan_indices[-2] + 1:
                    self.data[i][nan_indices[-1]] = (self.data[i][nan_indices[-1] - 1] + self.data[i][nan_indices[-1] + 1]) / 2
        
                # cast away missing & long segment
                nan_indices = np.where(np.isnan(self.data[i]))[0]
                nan_len = len(nan_indices)
                
                if nan_indices[0] >= pred_len + seq_len:
                    new_array_list.append(self.data[i][:nan_indices[0]])
                    idx_map.append(np.arange(idx_start, idx_start + nan_indices[0] - seq_len - pred_len + 1))
                    idx_start += nan_indices[0]
                for j in range(nan_len - 1):
                    if nan_indices[j + 1] - nan_indices[j] > pred_len + seq_len:
                        new_array_list.append(self.data[i][nan_indices[j] + 1:nan_indices[j + 1]])
                        idx_map.append(np.arange(idx_start, idx_start + nan_indices[j + 1] - nan_indices[j] - seq_len - pred_len))
                        idx_start += nan_indices[j + 1] - nan_indices[j] - 1
                if len(self.data[i]) - nan_indices[-1] > pred_len + seq_len:
                    new_array_list.append(self.data[i][nan_indices[-1] + 1:])
                    idx_map.append(np.arange(idx_start, idx_start + len(self.data[i]) - nan_indices[-1] - seq_len - pred_len))
                    idx_start += len(self.data[i]) - nan_indices[-1] - 1
           
        self.data = np.concatenate(new_array_list)
        self.idx_map = np.concatenate(idx_map)
        
        self.data = torch.from_numpy(self.data).to(device)
        if unsqueeze:
            self.data = torch.unsqueeze(self.data, dim=1)
            
        
    def __len__(self):
        return len(self.idx_map)
    
    def __getitem__(self, index):
        idx = self.idx_map[index]
        # if not self.unsqueeze:
        #     return torch.from_numpy(self.data[idx: idx + self.seq_len]),\
        #         torch.from_numpy(self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
        # else:
        #     return torch.from_numpy(self.data[idx: idx + self.seq_len].reshape(-1, 1)), \
        #         torch.from_numpy(self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len].reshape(-1, 1))
        
        return self.data[idx: idx + self.seq_len],\
            self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]

class UnivDatasetPlot(Dataset):
    def __init__(self, train_sets, seq_len, pred_len, train_prop=0.7, valid_prop=0.1, phase="train", unsqueeze=False) -> None:
        super().__init__()
        
        self.data, _ = read_sets(train_sets)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.unsqueeze = unsqueeze
        
        new_array_list = []
        idx_map = []
        idx_start = 0
        
        # for i in range(len(self.data)):
        for i in range(1):
            i = len(self.data) - 1
            # i = 1
            
            scaler = StandardScaler()
            
            train_data = self.data[i][:int(train_prop * len(self.data[i]))].reshape(-1, 1)
            scaler.fit(train_data)
            if phase == "train":
                self.data[i] = scaler.transform(train_data)
            elif phase == "valid":
                valid_data = self.data[i][int(train_prop * len(self.data[i])) - seq_len: int((train_prop + valid_prop) * len(self.data[i]))]
                self.data[i] = scaler.transform(valid_data.reshape(-1, 1))
            elif phase == "test":
                test_data = self.data[i][int((train_prop + valid_prop) * len(self.data[i])) - seq_len:]
                self.data[i] = scaler.transform(test_data.reshape(-1, 1))
            else:
                raise NotImplementedError("Wrong phase.")
            
            self.data[i] = self.data[i].flatten()
            
            array_len = len(self.data[i])
            
            # to small dataset
            if array_len < pred_len + seq_len:
                continue
            
            # fixing missing value
            nan_indices = np.where(np.isnan(self.data[i]))[0]
            nan_len = len(nan_indices)
            
            if nan_len == 0:
                new_array_list.append(self.data[i])
                idx_map.append(np.arange(idx_start, idx_start + len(self.data[i]) - seq_len - pred_len + 1))
                idx_start += len(self.data[i])
                
            else:
                # filling one point missing value
                if nan_indices[0] > 0 and nan_indices[1] != nan_indices[0] + 1:
                    self.data[i][nan_indices[0]] = (self.data[i][nan_indices[0] - 1] + self.data[i][nan_indices[0] + 1]) / 2
                for j in range(1, nan_len - 1):
                    if nan_indices[j - 1] != nan_indices[j] - 1 and nan_indices[j + 1] != nan_indices[j] + 1:
                        self.data[i][nan_indices[j]] = (self.data[i][nan_indices[j] - 1] + self.data[i][nan_indices[j] + 1]) / 2
                if nan_indices[-1] < array_len - 1 and nan_indices[-1] != nan_indices[-2] + 1:
                    self.data[i][nan_indices[-1]] = (self.data[i][nan_indices[-1] - 1] + self.data[i][nan_indices[-1] + 1]) / 2
        
                # cast away missing & long segment
                nan_indices = np.where(np.isnan(self.data[i]))[0]
                nan_len = len(nan_indices)
                
                if nan_indices[0] >= pred_len + seq_len:
                    new_array_list.append(self.data[i][:nan_indices[0]])
                    idx_map.append(np.arange(idx_start, idx_start + nan_indices[0] - seq_len - pred_len + 1))
                    idx_start += nan_indices[0]
                for j in range(nan_len - 1):
                    if nan_indices[j + 1] - nan_indices[j] > pred_len + seq_len:
                        new_array_list.append(self.data[i][nan_indices[j] + 1:nan_indices[j + 1]])
                        idx_map.append(np.arange(idx_start, idx_start + nan_indices[j + 1] - nan_indices[j] - seq_len - pred_len))
                        idx_start += nan_indices[j + 1] - nan_indices[j] - 1
                if len(self.data[i]) - nan_indices[-1] > pred_len + seq_len:
                    new_array_list.append(self.data[i][nan_indices[-1] + 1:])
                    idx_map.append(np.arange(idx_start, idx_start + len(self.data[i]) - nan_indices[-1] - seq_len - pred_len))
                    idx_start += len(self.data[i]) - nan_indices[-1] - 1
           
        self.data = np.concatenate(new_array_list)
        self.idx_map = np.concatenate(idx_map)
        
        self.data = torch.from_numpy(self.data).to(device)
        if unsqueeze:
            self.data = torch.unsqueeze(self.data, dim=1)
            
        
    def __len__(self):
        return len(self.idx_map)
    
    def __getitem__(self, index):
        idx = self.idx_map[index]
        # if not self.unsqueeze:
        #     return torch.from_numpy(self.data[idx: idx + self.seq_len]),\
        #         torch.from_numpy(self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
        # else:
        #     return torch.from_numpy(self.data[idx: idx + self.seq_len].reshape(-1, 1)), \
        #         torch.from_numpy(self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len].reshape(-1, 1))
        
        return self.data[idx: idx + self.seq_len],\
            self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]    

class UnivDecomDataset(Dataset):
    def __init__(self, train_sets, seq_len, pred_len, train_prop=0.7, valid_prop=0.1, phase="train", unsqueeze=False) -> None:
        super().__init__()
        
        self.data, self.p = read_sets(train_sets)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.unsqueeze = unsqueeze
        
        new_array_list = []
        period_list = []
        
        idx_map = []
        idx_start = 0
        
        for i in range(len(self.data)):
            scaler = StandardScaler()
            
            train_data = self.data[i][:int(train_prop * len(self.data[i]))].reshape(-1, 1)
            scaler.fit(train_data)
            if phase == "train":
                self.data[i] = scaler.transform(train_data)
            elif phase == "valid":
                valid_data = self.data[i][int(train_prop * len(self.data[i])) - seq_len: int((train_prop + valid_prop) * len(self.data[i]))]
                self.data[i] = scaler.transform(valid_data.reshape(-1, 1))
            elif phase == "test":
                test_data = self.data[i][int((train_prop + valid_prop) * len(self.data[i])) - seq_len:]
                self.data[i] = scaler.transform(test_data.reshape(-1, 1))
            else:
                raise NotImplementedError("Wrong phase.")
            
            self.data[i] = self.data[i].flatten()
            
            array_len = len(self.data[i])
            
            # to small dataset
            if array_len < pred_len + seq_len:
                continue
            
            # fixing missing value
            nan_indices = np.where(np.isnan(self.data[i]))[0]
            nan_len = len(nan_indices)
            
            if nan_len == 0:
                new_array_list.append(self.data[i])
                period_list.append([self.p[i]] * len(self.data[i]))
                
                idx_map.append(np.arange(idx_start, idx_start + len(self.data[i]) - seq_len - pred_len + 1))
                idx_start += len(self.data[i])
                
            else:
                raise RuntimeError("Finding missing values")
                # filling one point missing value
                if nan_indices[0] > 0 and nan_indices[1] != nan_indices[0] + 1:
                    self.data[i][nan_indices[0]] = (self.data[i][nan_indices[0] - 1] + self.data[i][nan_indices[0] + 1]) / 2
                for j in range(1, nan_len - 1):
                    if nan_indices[j - 1] != nan_indices[j] - 1 and nan_indices[j + 1] != nan_indices[j] + 1:
                        self.data[i][nan_indices[j]] = (self.data[i][nan_indices[j] - 1] + self.data[i][nan_indices[j] + 1]) / 2
                if nan_indices[-1] < array_len - 1 and nan_indices[-1] != nan_indices[-2] + 1:
                    self.data[i][nan_indices[-1]] = (self.data[i][nan_indices[-1] - 1] + self.data[i][nan_indices[-1] + 1]) / 2
        
                # cast away missing & long segment
                nan_indices = np.where(np.isnan(self.data[i]))[0]
                nan_len = len(nan_indices)
                
                if nan_indices[0] >= pred_len + seq_len:
                    new_array_list.append(self.data[i][:nan_indices[0]])
                    idx_map.append(np.arange(idx_start, idx_start + nan_indices[0] - seq_len - pred_len + 1))
                    idx_start += nan_indices[0]
                for j in range(nan_len - 1):
                    if nan_indices[j + 1] - nan_indices[j] > pred_len + seq_len:
                        new_array_list.append(self.data[i][nan_indices[j] + 1:nan_indices[j + 1]])
                        idx_map.append(np.arange(idx_start, idx_start + nan_indices[j + 1] - nan_indices[j] - seq_len - pred_len))
                        idx_start += nan_indices[j + 1] - nan_indices[j] - 1
                if len(self.data[i]) - nan_indices[-1] > pred_len + seq_len:
                    new_array_list.append(self.data[i][nan_indices[-1] + 1:])
                    idx_map.append(np.arange(idx_start, idx_start + len(self.data[i]) - nan_indices[-1] - seq_len - pred_len))
                    idx_start += len(self.data[i]) - nan_indices[-1] - 1
            
        self.data = np.concatenate(new_array_list)
        self.idx_map = np.concatenate(idx_map)
        
    def __len__(self):
        return len(self.idx_map)
    
    def __getitem__(self, index):
        idx = self.idx_map[index]
        # if not self.unsqueeze:
        #     return torch.from_numpy(self.data[idx: idx + self.seq_len]),\
        #         torch.from_numpy(self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
        # else:
        #     return torch.from_numpy(self.data[idx: idx + self.seq_len].reshape(-1, 1)), \
        #         torch.from_numpy(self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len].reshape(-1, 1))
        data = self.data[idx: idx + self.seq_len]
        period = self.p[idx]
        decomp = decompose(data, period)
        
        return decomp
    
    
class UnivDatasetFs(Dataset):
    def __init__(self, train_sets, seq_len, pred_len, train_prop=0.7, valid_prop=0.1, phase="train", unsqueeze=False, fs_prop=1) -> None:
        super().__init__()
        
        self.data, _ = read_sets(train_sets)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.unsqueeze = unsqueeze
        
        new_array_list = []
        idx_map = []
        idx_start = 0
        
        for i in range(len(self.data)):
            scaler = StandardScaler()
            
            train_data = self.data[i][:int(train_prop * len(self.data[i]))].reshape(-1, 1)
            scaler.fit(train_data)
            if phase == "train":
                self.data[i] = scaler.transform(train_data)
            elif phase == "valid":
                valid_data = self.data[i][int(train_prop * len(self.data[i])) - seq_len: int((train_prop + valid_prop) * len(self.data[i]))]
                self.data[i] = scaler.transform(valid_data.reshape(-1, 1))
            elif phase == "test":
                test_data = self.data[i][int((train_prop + valid_prop) * len(self.data[i])) - seq_len:]
                self.data[i] = scaler.transform(test_data.reshape(-1, 1))
            else:
                raise NotImplementedError("Wrong phase.")
            
            self.data[i] = self.data[i].flatten()
            
            array_len = len(self.data[i])
            
            # to small dataset
            if array_len < pred_len + seq_len:
                continue
            
            new_array_list.append(self.data[i])
            idx_map.append(np.arange(idx_start, idx_start + len(self.data[i]) - seq_len - pred_len + 1))
            idx_start += len(self.data[i])
            
           
        self.data = np.concatenate(new_array_list)
        self.idx_map = np.concatenate(idx_map)
        
        self.data = torch.from_numpy(self.data).to(device)
        if unsqueeze:
            self.data = torch.unsqueeze(self.data, dim=1)
        
        # build few sample set
        if phase == "train" and fs_prop != 1:
            np.random.shuffle(self.idx_map)
            self.idx_map = self.idx_map[:int(len(self.idx_map) * fs_prop)]
            
        self.len = len(self.idx_map)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        idx = self.idx_map[index]
        
        return self.data[idx: idx + self.seq_len],\
            self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
            
            
class UnivDatasetFs2(Dataset):
    def __init__(self, train_sets, seq_len, pred_len, train_prop=0.7, valid_prop=0.1, phase="train", unsqueeze=False, fs_prop=1) -> None:
        super().__init__()
        
        self.data, _ = read_sets(train_sets)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.unsqueeze = unsqueeze
        
        new_array_list = []
        idx_map = []
        idx_start = 0
        
        for i in range(len(self.data)):
            scaler = StandardScaler()
            
            train_data = self.data[i][int(train_prop * (1 - fs_prop) * len(self.data[i])):int(train_prop * len(self.data[i]))].reshape(-1, 1)
            scaler.fit(train_data)
            if phase == "train":
                self.data[i] = scaler.transform(train_data)
            elif phase == "valid":
                valid_data = self.data[i][int(train_prop * len(self.data[i])) - seq_len: int((train_prop + valid_prop) * len(self.data[i]))]
                self.data[i] = scaler.transform(valid_data.reshape(-1, 1))
            elif phase == "test":
                test_data = self.data[i][int((train_prop + valid_prop) * len(self.data[i])) - seq_len:]
                self.data[i] = scaler.transform(test_data.reshape(-1, 1))
            else:
                raise NotImplementedError("Wrong phase.")
            
            self.data[i] = self.data[i].flatten()
            
            array_len = len(self.data[i])
            
            # to small dataset
            if array_len < pred_len + seq_len:
                continue
            
            new_array_list.append(self.data[i])
            idx_map.append(np.arange(idx_start, idx_start + len(self.data[i]) - seq_len - pred_len + 1))
            idx_start += len(self.data[i])
            
           
        self.data = np.concatenate(new_array_list)
        self.idx_map = np.concatenate(idx_map)
        
        self.data = torch.from_numpy(self.data).to(device)
        if unsqueeze:
            self.data = torch.unsqueeze(self.data, dim=1)
            
        self.len = len(self.idx_map)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        idx = self.idx_map[index]
        
        return self.data[idx: idx + self.seq_len],\
            self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]