#!/usr/bin/python3

import sys
sys.path.append('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it')
import time
import rospy
import moveit_commander
import moveit_msgs.msg
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from math import pi
from control_msgs.msg import *
from trajectory_msgs.msg import *
import actionlib
from std_srvs.srv import Empty
from tf import TransformListener
# import gen3env
from robot import Robot
# from task import peg_in
# from gen3env import gen3env
import gym
# from stable_baselines3  import TD3
import torch.nn as nn
# from BC_model import BehaviorCloningModel
import torch
from Gen3Env.gen3env import gen3env
from stable_baselines3 import TD3
import os
import numpy as np

def run_model(model, initial_input, num_steps, frame, env):
    current_input = initial_input  # 初始输入

    for step in range(num_steps):
        # 将当前输入转换为 PyTorch 张量
        print(step)
        input_tensor = torch.Tensor(current_input)

        # 使用模型进行推断
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # 将输出添加到列表中
        action = output_tensor.tolist()
        action = origin_data(action)
        current_input = current_input[4:]
        # next_obs,reward,done,_,_=env.step(action=action)
        env.robot.move(pose=action[:3], tolerance=0.0001) # 前后 左右 上下
        env.robot.reach_gripper_position(action[-1])
        next_obs = env.get_obs()
        next_obs = normalize_data(np.array(next_obs[:4]))
        # 将模型的输出作为下一个步骤的输入
        if frame != 1:
            current_input = np.concatenate((current_input, next_obs))


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

# net
class BehaviorCloningModel(nn.Module):  # 搭建神经网络
    def __init__(self, input_dim, output_dim):
        super(BehaviorCloningModel, self).__init__()  # 继承自父类的构造
        self.fc = nn.Sequential(nn.Linear(input_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, output_dim)
                                )  # 搭建网络，两层隐藏层

    def forward(self, x):  # 前向传播方法
        return self.fc(x)

def main():
   env=gym.make(id='peg_in_hole-v0')
   env.reset()
   log_path='./log'
   if not os.path.exists(log_path):
      os.makedirs(log_path)
  #  print(torch.cuda.is_available())
   if torch.cuda.is_available():
      print('cuda is available, train on GPU!')
   model=TD3('MlpPolicy', env, verbose=1,tensorboard_log=log_path,device='cuda')
   model.learn(total_timesteps=1000)

def main2():
  # env=gym.make(id='Pendulum-v1')
  env=gym.make(id='peg_in_hole-v0')
  # env=gen3env()
  env.reset()
  model_path = '/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/model/model_epoch50000.pth'
  model = BehaviorCloningModel(16, 4)
  model.load_state_dict(torch.load(model_path))

  # # absolute
  # for episode in range(10):
  #   print("episode: {}".format(episode))
  #   env.robot.move(pose=[0.3, 0, 0.3])
  #   obs = env.reset()
  #   obs = np.array([[0.3, 0, 0.3, 0],[0.3, 0, 0.3, 0],[0.3, 0, 0.3, 0],[0.3, 0, 0.3, 0]])
  #   # print(type(obs))
  #   # print(obs)
  #   model_obs = obs[:4]
  #   done = False
  #   # while not done:
  #   for step in range(2):
  #     with torch.no_grad():
  #       action = model(torch.Tensor(model_obs)).tolist()
  #       # print(type(action))
  #     next_obs,reward,done,_=env.step(action=action)
  #     # print('reward={}'.format(reward))
  #     model_obs = next_obs[:4]
  # env.robot.move(pose=[0.3, 0, 0.3])

  for episode in range(10):
    print("episode: {}".format(episode))
    obs = env.reset()
    env.robot.move(pose=[0.3, 0, 0.3], tolerance=0.0001) # 前后 左右 上下
    env.robot.reach_gripper_position(0)
    with torch.no_grad():
        model_path = '/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/model/model_epoch50000.pth'
        model = BehaviorCloningModel(16, 4)
        model.load_state_dict(torch.load(model_path))
        initial_input = normalize_data(np.array([[ 2.99922740e-01, -3.85967414e-05,  2.99946854e-01,  2.65256679e-03],[2.99932234e-01, 1.24275835e-04, 2.99952932e-01, 4.80660593e-04],[2.99923132e-01, 6.25820728e-05, 2.99937792e-01, 3.69983390e-03],[2.99923132e-01, 6.25820728e-05, 2.99937792e-01, 3.69983390e-03]]))
        initial_input = initial_input.flatten()
        run_model(model, initial_input, num_steps=200, frame=4, env=env)



  # # delt
  # for episode in range(10):
  #   print("episode: {}".format(episode))
  #   env.robot.move(pose=[0.5, 0, 0.5])
  #   print("start")
  #   obs = env.reset()
  #   model_obs = obs[:4]
  #   done = False
  #   # while not done:
  #   for step in range(2):
  #     with torch.no_grad():
  #       action = model(torch.Tensor(model_obs)).tolist() + model_obs
  #     next_obs,reward,done,_=env.step(action=action)
  #     # print('reward={}'.format(reward))
  #     model_obs = next_obs[:4]
  # env.robot.move(pose=[0.5, 0, 0.5])
  
if __name__ == '__main__':
  main2()
