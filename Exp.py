import random
import time
import numpy as np
import argparse
import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import tqdm
from matplotlib import pyplot as plt
import scipy.stats as stats

from TSDataSets import *
from models import TSTPL
import toml
from torch.cuda.amp import autocast as autocast
import torchinfo

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('my_logger')

seed = 2024
# seed = 2024

if not seed:
    seed = random.randint(1, 10000)
print("seed is %d" %seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def set_logger(comments):
    folder_path = os.path.join("cpkt", comments)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_handler = logging.FileHandler('{}/log.txt'.format(folder_path))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(ch)

cuda = True
if cuda == True and torch.cuda.is_available():
    device = torch.device("cuda")
    print("=== Using CUDA ===")
else:
    if cuda == True and not torch.cuda.is_available():
        print("=== CUDA is unavailable ===")
    device = torch.device("cpu")
    print("=== Using CPU ===")


class Exp:
    def __init__(self) -> None:
        pass
    
    def build_model(self, model_name, seq_len, pred_len, llm_config, layer_idx, rope, num_hidden_layers=12, comments=None, requires_grad=True):
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        if comments is not None:
            self.save_path = os.path.join("cpkt", comments)
            self.comments = comments
        
        config = toml.load("config/{}.toml".format(model_name))["config"]
        args = argparse.Namespace(**config)
        
        args.llm_config = llm_config
        args.layer_idx = layer_idx
        args.seq_len = self.seq_len
        args.pred_len = self.pred_len
        args.requires_grad = requires_grad
        args.rope = rope
        args.num_hidden_layers = num_hidden_layers
        
        arg_dict = vars(args)
        with open('{}/config.toml'.format(self.save_path), 'w') as f:
            toml.dump(arg_dict, f)
        
        self.model = TSTPL.Model(args).to(device)
        self.mseloss = nn.MSELoss()
        
    def load_model(self, comment):
        config = toml.load('cpkt/{}/config.toml'.format(comment))
        args = argparse.Namespace(**config)
        self.model = TSTPL.Model(args).to(device)
        self.mseloss = nn.MSELoss()
        
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        
        
    def pre_train_phase(self, datasets, lr1, epochs, batch_size, use_amp=False, use_profiler=False, unsqueeze=False, it=-1):
        logger.info(">>> pre train phase, using dataset {}".format(datasets))
        
        train_loader = DataLoader(
            dataset=UnivDataset(datasets, self.seq_len, self.pred_len, phase="train", unsqueeze=unsqueeze),
            batch_size=batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=UnivDataset(datasets, self.seq_len, self.pred_len, phase="valid", unsqueeze=unsqueeze),
            batch_size=batch_size,
            shuffle=False
        )
        
        test_loaders = [
            DataLoader(
                dataset=UnivDataset([dataset], self.seq_len, self.pred_len, phase="test", unsqueeze=unsqueeze),
                batch_size=batch_size,
                shuffle=False
            ) for dataset in datasets
        ]
        
        self.pre_opt = optim.AdamW(self.model.parameters(), lr=lr1)
        self.pre_sch = optim.lr_scheduler.StepLR(self.pre_opt, step_size=3, gamma=0.75)
            
        for epoch in range(1, epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            train_loss, valid_loss, test_loss = 0,0,0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)

            if use_profiler:
                prof = torch.profiler.profile(
                        schedule=torch.profiler.schedule(wait=20, warmup=20, active=500, repeat=1),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler('./tracer/{}'.format(self.comments)),
                        record_shapes=True,
                        with_stack=True)
                prof.start()
            
            if use_amp:
                scaler = torch.cuda.amp.GradScaler()
            for idx, (x, target) in loop:
                
                if use_profiler:
                    prof.step()
                    if idx >= 20 + 20 + 500:
                        break
                
                x, target = x.float().to(device), target.float().to(device)
                self.pre_opt.zero_grad()
                
                if use_amp:
                    with autocast():
                        output = self.model(x)
                        loss = self.mseloss(output, target)
                        
                    scaler.scale(loss).backward()
                    
                    # scaler.unscale_(self.pre_opt) 
                    # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), 
                    #                                 max_norm=2.0,
                    #                                 norm_type=2,
                    #                                 error_if_nonfinite=False,)
                    
                    scaler.step(self.pre_opt)
                    scaler.update()
                        
                else:  
                    output = self.model(x)
                    loss = self.mseloss(output, target)
                    
                    loss.backward()
                    self.pre_opt.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                train_loss = avg_loss/(idx+1)
                
                if it != -1:
                    if idx % it == 0:
                        self.model.eval()
                        torch.save(self.model.state_dict(), "{}/it_{}.pth".format(self.save_path, idx))
                        self.model.train(mode=True)
                
            if use_profiler:
                prof.stop()
                break
            
            
            self.model.eval()
            if it == -1:
            # if epoch % 2 == 0:
                torch.save(self.model.state_dict(), "{}/{}.pth".format(self.save_path, epoch))
            else:
                break
                            
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)

            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.float().to(device), target.float().to(device)
                    
                    if use_amp:
                        with autocast():
                            output = self.model(x)
                            loss = self.mseloss(output, target)
                    else:
                        output = self.model(x)
                        loss = self.mseloss(output, target)
                        
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    valid_loss = avg_loss/(idx+1)
                    
            
            test_dict = {}
            for i, test_loader in enumerate(test_loaders):
                test_loss = 0
                avg_loss = 0
                
                y = []
                y_hat = []
                
                loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
                with torch.no_grad():
                    for idx, (x, target) in loop:
                        x, target = x.float().to(device), target.float().to(device)
                        if use_amp:
                            with autocast():
                                output = self.model(x)
                                loss = self.mseloss(output, target)
                                
                                y.append(target.flatten().detach().cpu().numpy())
                                y_hat.append(output.flatten().detach().cpu().numpy())
                        else:
                            output = self.model(x)
                            loss = self.mseloss(output, target)
                            
                            y.append(target.flatten().detach().cpu().numpy())
                            y_hat.append(output.flatten().detach().cpu().numpy())
                            
                        avg_loss += loss.cpu().item()
                        loop.set_description(f'Test Epoch [{epoch}/{epochs}]')
                        loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                y = np.concatenate(y)
                y_hat = np.concatenate(y_hat)
                test_loss = np.mean(np.square(y - y_hat))
                        
                test_dict[datasets[i]] = test_loss
            
            self.pre_sch.step()
            logger.info("[epoch: {}] train loss: {}, valid loss: {}, test loss: {}\n".format(epoch, train_loss, valid_loss, test_dict))
            
            
    def offline_test(self, model_path, dataset, batch_size, use_amp=False, unsqueeze=False):
        train_prop = 0.7
        valid_prop = 0.1
        
        # if dataset[:3] == "ETT":
        #     train_prop = 0.6
        #     valid_prop = 0.2
            
        test_loader = DataLoader(
            dataset=UnivDataset([dataset], self.seq_len, self.pred_len, phase="test", unsqueeze=unsqueeze, train_prop=train_prop, valid_prop=valid_prop),
            batch_size=batch_size,
            shuffle=False
        )
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        test_loss = 0
        avg_loss = 0
        
        y = []
        y_hat = []
        
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.float().to(device), target.float().to(device)
                if use_amp:
                    with autocast():
                        output = self.model(x)
                        y.append(target.flatten().detach().cpu().numpy())
                        y_hat.append(output.flatten().detach().cpu().numpy())
                        
                        loss = self.mseloss(output, target)
                else:
                    output = self.model(x)
                    loss = self.mseloss(output, target)
                    
                    y.append(target.flatten().detach().cpu().numpy())
                    y_hat.append(output.flatten().detach().cpu().numpy())
                    
                avg_loss += loss.cpu().item()
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                
        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)
        mae = np.mean(np.abs(y - y_hat))
        mse = np.mean(np.square(y - y_hat))
        
        print("dataset is {}, mse is {}, mae is {}".format(dataset, mse, mae))
        return test_loss
    
    def plot_samples(self, model_path, dataset, batch_size, use_amp=False, unsqueeze=False, comment=None):
        train_prop = 0.7
        valid_prop = 0.1
        
        # if dataset[:3] == "ETT":
        #     train_prop = 0.6
        #     valid_prop = 0.2
            
        test_loader = DataLoader(
            dataset=UnivDataset([dataset], self.seq_len, self.pred_len, phase="test", unsqueeze=unsqueeze, train_prop=train_prop, valid_prop=valid_prop),
            batch_size=batch_size,
            shuffle=False
        )
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        test_loss = 0
        avg_loss = 0
        
        y = []
        y_hat = []
        x_ = []
        
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.float().to(device), target.float().to(device)
                if use_amp:
                    with autocast():
                        output = self.model(x)
                        y.append(target.flatten().detach().cpu().numpy())
                        y_hat.append(output.flatten().detach().cpu().numpy())
                        
                        loss = self.mseloss(output, target)
                else:
                    output = self.model(x)
                    loss = self.mseloss(output, target)
                    
                    x_.append(x.flatten().detach().cpu().numpy())
                    y.append(target.flatten().detach().cpu().numpy())
                    y_hat.append(output.flatten().detach().cpu().numpy())
                    
                avg_loss += loss.cpu().item()
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                
        x_ = np.concatenate(x_)
        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)
        test_loss = np.mean(np.square(y - y_hat))
            
        
        
        print("dataset is {}, loss is {}".format(dataset, test_loss))
        
    def print_model_summary(self):
        logger.info(torchinfo.summary(self.model, input_size=(1, self.seq_len)))
        
            
if __name__ == "__main__":
    with open("info.json", "r") as f:
        datasets = json.load(f)
        
    
    seq_len = 512
    pred_len = 96
    pre_train_ds = datasets.keys()
    pre_train_ds = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic", "weather"]
    pre_lr = 0.0002 # setting for default
    epochs = 6
    batch_size = 256
    llm_config = "gemma-2b"
    # llm_config = "llama-3-8b"
    layer_idx = -1
    requires_grad = False
    rope = False
    num_hidden_layers = 12
    
    
    '''
    r: rope
    l: lookback
    p: predict window
    lc: llm_config
    rg: require grad
    li: layer idx
    '''
    
    add_com = "stride16"
    
    llm_config = "gemma-2b"
    comments = "PErMformer"
    
    
    set_logger(comments)
    
    exp = Exp()
    exp.build_model("TSTPL", seq_len, pred_len, llm_config, layer_idx, rope, num_hidden_layers, comments, requires_grad)
    logger.info(exp.print_model_summary())
    exp.pre_train_phase(pre_train_ds, pre_lr, epochs, batch_size, use_amp=True, use_profiler=False)
    # report the results with the minimum validation loss. 