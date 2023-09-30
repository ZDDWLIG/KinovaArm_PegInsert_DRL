import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
import os

# from readdata import read_all_data
# from plot import plot_np, plot_all, draw_np, plot_4dnp

import glob

import argparse

import queue
import random


def read_all_data(csv_dir):
        # 获取目录中所有CSV文件的文件名
    csv_files = glob.glob(csv_dir + "*.csv")
    print(csv_files)
    
    # 创建一个空的DataFrame，用于存储所有CSV文件的数据
    combined_data = pd.DataFrame()
    
    # 循环读取每个CSV文件并将其合并到combined_data中
    for file in csv_files:
        # 使用read_csv函数读取CSV文件
        data = pd.read_csv(file, header=None)
        # print(data)
        # 将数据追加到combined_data中
        combined_data = pd.concat([combined_data, data], axis=0, ignore_index=True)
    
    # combined_data包含了所有CSV文件的数据，每个CSV文件的行数仍然保持不变
    return combined_data


#距离过小或夹爪变化小于 gripper_change_deta 删除本组数据
def data_processing(state, delt, gripper_change_delt):
    p = state[0,:]
    delete_line = []
    for i in range(1,state.shape[0]):
        distance = np.sum(np.square(p[:3]-state[i][:3]))  
        gripper_change = abs(p[3]-state[i][3])
        
        if(gripper_change <= gripper_change_delt and distance < delt):
            delete_line.append(i)
        else:
            p = state[i]
    # processed_data = np.delete(state,delete_line,0)
    return delete_line

def normalize_data(data):
    min_vals = np.array([0.299900302, -0.17102845, 0.05590736, -0.000087572115])
    max_vals = np.array([0.58015204, 0.00020189775, 0.299989649, 0.36635616])
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def origin_data(data):
    min_vals = np.array([0.299900302, -0.17102845, 0.05590736, -0.000087572115])
    max_vals = np.array([0.58015204, 0.00020189775, 0.299989649, 0.36635616])
    origin_data_data = (max_vals - min_vals) * data + min_vals
    return origin_data_data

def save_parameters_to_txt(log_dir, **kwargs):
    # os.makedirs(log_dir)
    filename = os.path.join(log_dir, "log.txt")
    with open(filename, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

# 叠加帧
def overlay_frames(state, action, frame_length):
    inputs = []
    outputs = []
    for i in range(len(state) - frame_length):
        input_sequence = state[i:(i + frame_length)].flatten()
        output_sequence = state[i + frame_length]  # 使用npaction的第四行作为输出
        inputs.append(input_sequence)
        outputs.append(output_sequence)
    return np.array(inputs), np.array(outputs)



 
class ExpertLoader():
    def __init__(self, nstep, batch):
        self.nstep = nstep
        self.batch=batch
        csv_dir = "/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it_zuo/expert_data/new_hole_optimize2/"
        self.data = read_all_data(csv_dir)
        self.data_list = []
        self.data_list.append(self.read_data(0, 1))
        self.data_list.append(self.read_data(2, 3))
        self.data_list.append(self.read_data(6, 7))
        self.data_storage = []
        for data in self.data_list:
            state, action = data
            seq_queue = queue.Queue(self.nstep)
            for i in range(self.nstep):
                seq_queue.put(state[0])
            for i in range(len(data)):
                seq_queue.get()
                seq_queue.put(state[i])
                self.data_storage.append((np.stack(seq_queue.queue), action[i]))

    def read_data(self, stateid, actionid):
        state = self.data.iloc[stateid].to_numpy()
        npstate = np.array([np.fromstring(item[1:-1], sep=' ')
                        for item in state if not isinstance(item, float)])  # 将state变为(?, 4)的格式，一行代表一个state
        action = self.data.iloc[actionid].to_numpy()
        npaction = np.array([np.fromstring(item[1:-1], sep=' ') 
                             for item in action if not isinstance(item, float)])
        npstate[:,:4] = normalize_data(npstate[:,:4])
        npaction = normalize_data(npaction)
        delt = 1e-3
        gripper_change_delt = 0.02/(0.36635616+0.000087572115)
        del_state = data_processing(npstate, delt, gripper_change_delt)
        # del_action = data_processing(npaction, delt, gripper_change_delt)
        del_action = []
        del_lines = list(set(del_state + del_action))
        npstate = np.delete(npstate,del_lines,0)
        npaction = np.delete(npaction,del_lines,0)
        npstate[:,:4] = origin_data(npstate[:,:4])
        npaction = origin_data(npaction)
        # action[0] = (action[0] + 1) * 0.2 + 0.2
        # action[1] = (action[1] + 1) * -0.075
        # action[2] = (action[2] + 1) * 0.125 + 0.05
        # action[3] = (action[3] + 1) * 0.2
        npaction[:,0] = (npaction[:,0] - 0.2) / 0.2 - 1.3
        npaction[:,1] = (npaction[:,1] / -0.1) / 1.1 - 1
        npaction[:,2] = (npaction[:,2] - 0.05) / 0.125 - 0.75
        npaction[:,3] = (npaction[:,3] / 0.2) - 1
        return npstate, npaction
        
    def __iter__(self):
        while True:
            samples = random.choices(self.data_storage, k=self.batch)
            # print([i[0].shape for i in samples])
            samples = [np.stack([x[i] for x in samples]) for i in range(len(samples[0]))]
            # [batch, len, obs] [batch, act]
            yield samples
    