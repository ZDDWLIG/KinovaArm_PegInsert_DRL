#!/usr/bin/python3

import sys
sys.path.append('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it')
sys.path.extend(['', '/catkin_workspace/devel/lib/python3/dist-packages', '/opt/ros/noetic/lib/python3/dist-packages', '/home/user/miniconda/lib/python38.zip', '/home/user/miniconda/lib/python3.8', '/home/user/miniconda/lib/python3.8/lib-dynload', '/home/user/.local/lib/python3.8/site-packages', '/home/user/miniconda/lib/python3.8/site-packages'])
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
from robot import Robot
# from task import peg_in
# from gen3env import gen3env as RobotEnv
import gym
import torch.nn as nn
from network import MLPModel, LSTMModel
import torch
from Gen3Env.gen3env import gen3env
# from stable_baselines3 import TD3
import os
import numpy as np
# from stable_baselines3 import TD3
# from module import TD3,train
from stable_baselines3 import TD3
import subprocess

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

def rl_train():
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
def test():
    # init_gazebo()
    # os.system('python /catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/test_os.py')
    # os.system('cd /catkin_workspace && roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85')
    # p=
    # p=subprocess.run('roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #  def remove_scene(self):
    # try:
    #   self.delete_model('peg')
    #   self.delete_model('hole')
    # except Exception as e:
    #   # print('peg', e)
    #   print('Fail to delete, start to reset')
    #   self.reset_scene()
    env=gym.make(id='peg_in_hole-v0')
    # env.robot.reset_scene()
    # env.reset()
    if torch.cuda.is_available():
        print('Use GPU to train!')
    else:
        print('Could not find GPU,lets use CPU!')
    model_path='/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/model/model_results'
    model=TD3('MlpPolicy',env=env,verbose=1)
    model=TD3.load('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/model/model_results_34.zip')
    model.set_env(env=env)
    # # model.load()
    for i in range(100):
        model.learn(total_timesteps=50,reset_num_timesteps=False)
        model.save(model_path+f'_{i}')
    
    
    
    # action=env.action_space.sample()
    # obs=env.get_obs()
    # dis=env.
    
    
def bc_run():
    # env=gym.make(id='Pendulum-v1')
    model_name = 'lstm'
    run_env = 'env'
    
    frame = 6
    episodes = 10
    steps = 200
    model_path = '/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/model/model_epoch40000.pth'

    if run_env == 'env':
        env=gym.make(id='peg_in_hole-v0')
    if run_env == 'robot':
        env=RobotEnv()

    if model_name == 'mlp':
        model = MLPModel(frame*4, 4)
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

        # new absolute
        for episode in range(episodes):
            print("episode: {}".format(episode))
            obs = env.reset()
            env.robot.move(pose=[0.3, 0, 0.3], tolerance=0.0001) # 前后 左右 上下
            env.robot.reach_gripper_position(0)
            obs = env.get_obs()
            obs = normalize_data(np.array(obs[:4]))
            obs = np.tile(obs, (frame, 1))
            obs = obs.flatten()
            with torch.no_grad():
                for step in range(steps):
                    print(f'step: {step}')
                    input_tensor = torch.Tensor(obs)
                    output_tensor = model(input_tensor)
                    action = output_tensor.tolist()
                    action = origin_data(action)
                    
                    env.robot.move(pose=action[:3])
                    env.robot.reach_gripper_position(action[-1])
                    
                    obs = obs[4:]
                    next_obs = env.get_obs()
                    next_obs = normalize_data(np.array(next_obs[:4]))

                    if frame !=1:
                        obs = np.concatenate((obs, next_obs))

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
  
    if model_name == 'lstm':
        model = LSTMModel(4, 64, 4)
        model.load_state_dict(torch.load(model_path))

        for episode in range(episodes):
            print(f"episode: {episode}")
            obs = env.reset()
            env.robot.move(pose=[0.3, 0, 0.3], tolerance=0.0001) # 前后 左右 上下
            env.robot.reach_gripper_position(0)
            obs = env.get_obs()
            # print(obs)
            obs = normalize_data(np.array(obs[:4]))

            with torch.no_grad():
                hidden = None
                for step in range(steps):
                    print(f'step: {step}')
                    input_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    output_tensor, hidden = model(input_tensor, hidden) # (1,1,4)
                    action = output_tensor.squeeze().squeeze() # (4,)
                    action = action.tolist()
                    action = origin_data(action)

                    env.robot.move(pose=action[:3])
                    env.robot.reach_gripper_position(action[-1])

                    next_obs = env.get_obs()
                    next_obs = normalize_data(np.array(next_obs[:4]))

                    obs = next_obs

# if __name__ == '__main__':
    # test()



import agent.potil_lstm as agent
import agent.utils as utils
# import fake_env
from logger import Logger
from pathlib import Path
import queue
import random


import numpy as np
import torch
from expert_dataloader import ExpertLoader

torch.backends.cudnn.benchmark = True

potil_agent = agent.POTILAgent(obs_shape=[10, 1], action_shape=[4, 1], device='cuda', lr=1e-5, feature_dim=512,
                               hidden_dim=128, critic_target_tau=0.01, num_expl_steps=0,
                               update_every_steps=2, stddev_schedule='linear(0.2,0.04,500000)', stddev_clip=0.1, use_tb=True, augment=True,
                               rewards='sinkhorn_cosine', sinkhorn_rew_scale=200, update_target_every=10000,
                               auto_rew_scale=True, auto_rew_scale_factor=10, obs_type='env', bc_weight_type='linear',
                               bc_weight_schedule='linear(0.15,0.03,20000)')



class ReplayBuffer:
    def __init__(self, batch, max_n):
        self.batch = batch
        self.max_n = max_n
        self.buffer = queue.Queue(max_n)

    def add(self, timestep):
        if self.buffer.qsize() == self.max_n:
            self.buffer.get()
        self.buffer.put(timestep)

    def sample(self):
        samples = random.choices(self.buffer.queue, k=self.batch)
        return [np.stack([x[i] for x in samples]) for i in range(len(samples[1]))]

    def empty(self):
        return self.buffer.empty()

    def __iter__(self):
        while True:
            yield self.sample()

    def __len__(self):
        return len(self.buffer.queue)


# class ExpertLoader:
#     def __iter__(self):
#         while True:
#             # obs actor
#             yield (np.random.randn(10, 10, 4), np.random.randn(10, 4))


class Trainer():
    def __init__(self):
        self.num_train_frames = 2100000
        self.action_repeat = 1
        self.num_seed_frames = 0
        self.eval_every_frames = 20000
        self.bc_regularize = True
        self.work_dir = Path('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/rot_work_dir')
        self.obs_type = 'features'
        self.num_demos = 10
        self.batch_size = 512
        self.max_seq_len = 1
        self.train_env = gym.make(id='peg_in_hole-v0')
        self.agent = potil_agent
        self._global_step = 0
        self._global_episode = 0
        self.timer = utils.Timer()
        self.logger = Logger(self.work_dir, use_tb=True)
        self.expert_replay_loader = ExpertLoader(self.max_seq_len, self.batch_size) # obs_bc, action_bc
        # self.expert_demo = np.random.randn(5, 350, 4)
        self.expert_demo = np.expand_dims(self.expert_replay_loader.data_list[0][0], 0)
        # self.expert_reward = np.random.randn(200)
        self.expert_replay_iter = iter(self.expert_replay_loader)
        self._replay_iter = None
        self.replay_loader = ReplayBuffer(self.batch_size, 1.5e5) # obs, action, reward, discount, next_obs
        # self.load_snapshot(self.work_dir / 'snapshot.pt')


    def train(self):
        # predicates
        train_until_step = utils.Until(self.num_train_frames,
                                       self.action_repeat)
        seed_until_step = utils.Until(self.num_seed_frames,
                                      self.action_repeat)

        episode_step, episode_reward = 0, 0

        time_steps = list()
        observations = list()
        actions = list()
        rewards = list()
        seq_queue = queue.Queue(self.max_seq_len)

        time_step = [self.train_env.reset()]
        # obs
        time_steps.append(time_step)
        for i in range(self.max_seq_len):
            seq_queue.put(time_step[0])
        obs = np.stack(seq_queue.queue)
        observations.append(obs)


        if repr(self.agent) == 'potil':
            if self.agent.auto_rew_scale:
                self.agent.sinkhorn_rew_scale = 1.  # Set after first episode

        metrics = None
        done = False
        while train_until_step(self.global_step):
            if done:
                self._global_episode += 1
                # # wait until all the metrics schema is populated
                # observations = np.stack(observations, 0)
                # actions = np.stack(actions, 0)
                # if repr(self.agent) == 'potil':
                #     new_rewards = self.agent.ot_rewarder(
                #         observations, self.expert_demo, self.global_step)
                #     new_rewards_sum = np.sum(new_rewards)
                # elif repr(self.agent) == 'dac':
                #     new_rewards = self.agent.dac_rewarder(observations, actions)
                #     new_rewards_sum = np.sum(new_rewards)
                #
                # if repr(self.agent) == 'potil':
                #     if self.agent.auto_rew_scale:
                #         if self._global_episode == 1:
                #             self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(
                #                 np.abs(new_rewards_sum))
                #             new_rewards = self.agent.ot_rewarder(
                #                 observations, self.expert_demo, self.global_step)
                #             new_rewards_sum = np.sum(new_rewards)


                for i, elt in enumerate(time_steps):
                    # obs,reward,done,truncated,info
                    # obs, action, reward, discount, next_obs
                    if i == len(time_steps) - 1:
                        break
                    reward = rewards[i]
                    # if repr(self.agent) == 'potil' or repr(self.agent) == 'dac':
                    #     reward = new_rewards[i]
                    storage = (observations[i], actions[i], np.array(reward, dtype=np.float32), np.array(0.99, dtype=np.float32), observations[i + 1])
                    self.replay_loader.add(storage)


                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.action_repeat
                    print('episode_reward', episode_reward)
                    # with self.logger.log_and_dump_ctx(self.global_frame,
                                                    #   ty='train') as log:
                        # log('fps', episode_frame / elapsed_time)
                        # log('total_time', total_time)
                        # log('episode_reward', episode_reward)
                        # log('episode_length', episode_frame)
                        # log('episode', self.global_episode)
                        # log('buffer_size', len(self.replay_loader))
                        # log('step', self.global_step)
                        # if repr(self.agent) == 'potil' or repr(self.agent) == 'dac':
                        #     log('expert_reward', np.sum(self.expert_reward))
                            # log('imitation_reward', new_rewards_sum)

                # reset env
                time_steps = list()
                observations = list()
                actions = list()
                rewards = list()
                seq_queue = queue.Queue(self.max_seq_len)

                time_step = [self.train_env.reset()]
                done = False
                time_steps.append(time_step)
                for i in range(self.max_seq_len):
                    seq_queue.put(time_step[0])
                obs = np.stack(seq_queue.queue)
                observations.append(obs)
                # try to save snapshot
                if self.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0


            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(obs.astype(np.float32),
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step) and not self.replay_loader.empty():
                # Update
                metrics = self.agent.update(self.replay_iter, self.expert_replay_iter,
                                            self.global_step, self.bc_regularize)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            # obs,reward,done,truncated,info
            done = bool(time_step[2])
            episode_reward += time_step[1]

            if seq_queue.qsize() == self.max_seq_len:
                seq_queue.get()
            seq_queue.put(time_step[0])
            obs = np.stack(seq_queue.queue)
            time_steps.append(time_step)
            observations.append(obs)
            actions.append(action)
            rewards.append(time_step[1])

            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with open(snapshot, 'wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        print(agent_payload)
        self.agent.load_snapshot(agent_payload)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

def test_gripper():
    env=gym.make('peg_in_hole-v0')
    env.robot.move(pose=[0.4,0.2,0.3])
    env.robot.reach_gripper_position(0.8)
    env.reset()

def test_bc():
    # potil_agent.actor = torch.load('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/BC_actor_dist.pkl')
    # nstep = 5
    # env=gym.make('peg_in_hole-v0')
    # seq_queue = queue.Queue(nstep)
    # env.reset()
    # env.robot.move(pose=[0.3, 0, 0.3], tolerance=0.0001)
    # expert_loader = ExpertLoader(nstep, 50)
    # test_data = expert_loader.read_data(0,1)
    # for i in range(nstep):
    #     action = test_data[1][i]
    #     time_step = env.step(action)
    #     seq_queue.put(time_step[0])
    
    # # obs
    # # for i in range(nstep):
    # #     seq_queue.put(time_step[0])
    # obs = np.stack(seq_queue.queue)
    # for i in range(80):
    #     with torch.no_grad(), utils.eval_mode(potil_agent):
    #         action = potil_agent.act(obs.astype(np.float32),i,eval_mode=True)
    #     time_step = env.step(action)
    #     # obs,reward,done,truncated,info
    #     done = bool(time_step[2])
    
    #     if done:
    #         break
    #     if seq_queue.qsize() == nstep:
    #         seq_queue.get()
    #     seq_queue.put(time_step[0])
    #     obs = np.stack(seq_queue.queue)

    potil_agent.actor = torch.load('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/BC_actor_dist.pkl')
    env=gym.make('peg_in_hole-v0')
    env.reset()
    env.robot.move(pose=[0.3, 0, 0.3], tolerance=0.0001)
    obs = env.get_obs()
    hidden = None
    for i in range(80):
        with torch.no_grad(), utils.eval_mode(potil_agent):
            # action = potil_agent.act(obs.astype(np.float32),i,eval_mode=True)
            action, hidden = potil_agent.act_with_hid(obs.astype(np.float32)[:4], hidden)
        time_step = env.step(action)
        # obs,reward,done,truncated,info
        done = bool(time_step[2])
        if done:
            break
        obs = time_step[0]
    


def test_expert():
    nstep = 10
    expert_loader = ExpertLoader(nstep, 50)
    expert_iter = iter(expert_loader)
    test_data = expert_loader.read_data(0,1)
    state, action = test_data
    seq_queue = queue.Queue(nstep)
    test_data_storage = []
    for i in range(nstep):
        seq_queue.put(state[0])
    for i in range(state.shape[0]):
        seq_queue.get()
        seq_queue.put(state[i])
        test_data_storage.append((np.stack(seq_queue.queue), action[i]))

    env=gym.make('peg_in_hole-v0')
    seq_queue = queue.Queue(nstep)
    time_steps = list()
    time_step = [env.reset()]
    # obs
    time_steps.append(time_step)
    for i in range(nstep):
        seq_queue.put(time_step[0])
    obs = np.stack(seq_queue.queue)

    for data in test_data_storage:
        action = data[1]
        time_step = env.step(action)
        # obs,reward,done,truncated,info
        done = bool(time_step[2])
        if done:
            break
        if seq_queue.qsize() == nstep:
            seq_queue.get()
        seq_queue.put(time_step[0])
        obs = np.stack(seq_queue.queue)
    
        


if __name__ == '__main__':
    test_expert()

    # test_bc()

    # test_gripper()

    # potil_agent.actor = torch.load('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/BC_actor_dist.pkl')
    # trainer = Trainer()
    # trainer.train()

    
    # env=gym.make(id='peg_in_hole-v0')
    # env.reset()
    # peg_pos=env.robot.get_obj_pose("peg")
    # hole_pos=env.robot.get_obj_pose("hole")
    # print(peg_pos)
    # print(hole_pos)

