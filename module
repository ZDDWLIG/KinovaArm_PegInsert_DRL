import torch as th
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque,namedtuple
from torch.utils.tensorboard import SummaryWriter
import os
from matplotlib import pyplot as plt
device=th.device('cuda' if th.cuda.is_available() else 'cpu')
class ReplayBuffer():
    def __init__(self,mem_size,batch_size):
        self.mem_size=mem_size
        self.batch_size=batch_size
        self.memory=deque(maxlen=self.mem_size)
        self.experience=namedtuple(typename='experience',field_names=['state','action','reward','next_state','done'])
    def add(self,state,action,reward,next_state,done):
        experience=self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    def __len__(self):
        return len(self.memory)
    def sample(self):
        experiences=random.sample(self.memory,k=self.batch_size)
        states=th.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions=th.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards=th.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states=th.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones=th.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return states,actions,rewards,next_states,dones

def min_q(x,y,dim=1):
    z=th.stack([x,y],dim=dim)
    min_z,_=th.min(z,dim=dim)
    return min_z



class Actor(nn.Module):
    def __init__(self,  action_dim,state_dim, max_action, drop=False):
        super(Actor, self).__init__()
        self.max_action = max_action
        if drop:
            self.mlp=nn.Sequential(nn.Linear(state_dim,512),nn.ReLU(),nn.Dropout(0.1),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,action_dim))
        else:
            self.mlp=nn.Sequential(nn.Linear(state_dim,512),nn.ReLU(),nn.Dropout(0),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,action_dim))
    def forward(self, state):
        x = self.max_action * th.tanh(self.mlp(state))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.mlp=nn.Sequential(nn.Linear(state_dim + action_dim, 512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1))
    def forward(self, state, action):
        x = self.mlp(th.cat([state, action], dim=1))
        return x
class TD3():
    def __init__(self,env,batch_size=128,lr=1e-3,
                 gamma=0.99,noise=0.1,mem_size=int(1e6),tau=5e-3,tar_update=5,policy_update=1,sigma=0.2,c=0.5):
        self.action_dim=env.action_space.shape[0]
        self.state_dim=env.observation_space.shape[0]
        self.min_action=env.action_space.low[0]
        self.max_action=env.action_space.high[0]
        self.batch_size=batch_size
        self.lr=lr
        self.gamma=gamma
        self.noise=noise
        self.mem_size=mem_size
        self.tau=tau
        self.tar_update=tar_update
        self.policy_update=policy_update
        self.sigma=sigma
        path=os.path.abspath(__file__)
        path='/'.join(path.split('/')[:-1])
        # self.model_path=path+'/results/TD3/checkpoints/'
        self.log_path=path+'/results/TD3/logs/'
        self.log_writer=SummaryWriter(self.log_path)
        self.c=c
        self.actor_net=Actor(self.action_dim,self.state_dim,self.max_action,True).to(device)
        self.actor_tar_net=Actor(self.action_dim,self.state_dim,self.max_action).to(device)
        self.critic_net1=Critic(self.action_dim,self.state_dim).to(device)
        self.critic_tar_net1=Critic(self.action_dim,self.state_dim).to(device)
        self.critic_net2 = Critic(self.action_dim, self.state_dim).to(device)
        self.critic_tar_net2 = Critic(self.action_dim, self.state_dim).to(device)
        self.actor_tar_net.load_state_dict(self.actor_net.state_dict())
        self.critic_tar_net1.load_state_dict(self.critic_net1.state_dict())
        self.critic_tar_net2.load_state_dict(self.critic_net2.state_dict())
        self.replay_buffer=ReplayBuffer(self.mem_size,self.batch_size)
        self.loss=nn.MSELoss()
        self.actor_net_opt=optim.Adam(self.actor_net.parameters(),1e-4)
        self.critic_net1_opt=optim.Adam(self.critic_net1.parameters(),1e-3)
        self.critic_net2_opt=optim.Adam(self.critic_net2.parameters(),1e-3)
        self.memory=ReplayBuffer(self.mem_size,self.batch_size)
        self.log_q_file = (open(f'{self.log_path}/log_q_file_0.txt', 'w'), 0)
    def tar_net_update(self):
        for eval_param, target_param in zip(self.actor_net.parameters(), self.actor_tar_net.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)
        for eval_param, target_param in zip(self.critic_net1.parameters(), self.critic_tar_net1.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)
        for eval_param, target_param in zip(self.critic_net2.parameters(), self.critic_tar_net2.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

    def action_sample(self,state):
        state=th.from_numpy(state).float().to(device)
        action=self.actor_net(state).cpu().data.numpy().flatten()
        action=(action+np.random.normal(0,self.noise,size=self.action_dim)).clip(self.min_action,self.max_action)
        return action

    def learn(self,experiences,eps,step,tot_step):
        states, actions, rewards, next_states, dones=experiences
        
        #updata critic netwark with TD
        next_actions=self.actor_tar_net(next_states)
        action_noise=th.normal(mean=0.,std=self.sigma,size=next_actions.size()).clip(-self.c,self.c).to(device)#add noise
        next_actions=(next_actions+action_noise).clip(self.min_action,self.max_action)
        tar_q=min_q(self.critic_tar_net1(next_states,next_actions),self.critic_tar_net2(next_states,next_actions),dim=1)
        y_t=rewards+self.gamma*(1-dones)*tar_q.detach()#target net dont need to backward, we have "tar_net_updata()"
        #updtae critic1
        eval_q1=self.critic_net1(states,actions)
        critic_l1=self.loss(eval_q1,y_t)
        self.critic_net1_opt.zero_grad()
        critic_l1.backward()
        self.critic_net1_opt.step()
        #update critic2
        eval_q2=self.critic_net2(states,actions)
        critic_l2 = self.loss(eval_q2, y_t)
        self.critic_net2_opt.zero_grad()
        critic_l2.backward()
        self.critic_net2_opt.step()
        #updata actor network with DPG
        if eps%self.policy_update==0:
            actor_l=-self.critic_net1(states,self.actor_net(states)).mean()
            self.actor_net_opt.zero_grad()
            actor_l.backward()
            self.actor_net_opt.step()
            self.log_writer.add_scalars('loss',{'actor_loss':actor_l.item(),
                                            'critic_loss':critic_l1.item()},tot_step)
            if self.log_q_file[1] != eps:
                self.log_q_file = (open(f'{self.log_path}/log_q_file_{eps}.txt', 'w'), eps)
            print(f'{list(eval_q1.squeeze().detach().cpu().numpy())} {list(eval_q2.squeeze().detach().cpu().numpy())}', file=self.log_q_file[0])
            self.log_q_file[0].flush()

        #updata target network
        if eps%self.tar_update==0:
            self.tar_net_update()
    def train(self,state,action,reward,next_state,done,eps,step, tot_step):
        self.memory.add(state,action,reward,next_state,done)
        if len(self.memory)>self.memory.batch_size:
            experiences=self.memory.sample()
            self.learn(experiences,eps,step, tot_step)


def train(env,agent,episodes=50000,save_inter=500,max_step=200,target_score=250,path=os.path.abspath(__file__),save_model=True):
    if th.cuda.is_available():
        print('Lets use gpu!')
    else:
        print('Lets use cpu!')
    score_list=[]
    flag=False
    path='/'.join(path.split('/')[:-1])
    model_path=path+'/results/TD3/checkpoints/'
    log_path=path+'/results/TD3/logs/'
    log_writer = SummaryWriter(log_path)
    tot_step = 0
    for ep in range(episodes):
        score=0.

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        state=env.reset()
        for step in range(max_step):
            action=agent.action_sample(state)
            next_state,reward,done,info,_=env.step(action)
            agent.train(state,action,reward,next_state,done,ep,step, tot_step)
            tot_step += 1
            if done:
                break
            state=next_state
            score+=reward
        score_list.append(score)
        score_avg=np.mean(score_list[-100:])
        log_writer.add_scalar('score_avg',score,ep)
        if ep%save_inter==0:
            if save_model:
                th.save(agent.actor_net.state_dict(),model_path+f'model_{ep}_actor.pth')
                th.save(agent.critic_net1.state_dict(),model_path+f'model_{ep}_critic1.pth')
                th.save(agent.critic_net2.state_dict(),model_path+f'model_{ep}_critic2.pth')
            print('episode={},average_score={}'.format(ep,score_avg))
        if len(score_list)>-130:
            if score>target_score:
                flag=True
                break
    if flag:
        print('Target achieved!')
    else:
        print('Episodes is too small, target is not achieved...')



def test(env,agent,test_episodes=5,max_step=200):
    score_list=[]
    for ep in range(test_episodes):
        state,_=env.reset()
        score=0.
        for step in range(max_step):
            action=agent.action_sample(state)
            next_state,reward,done,info,_=env.step(action)
            state=next_state
            score+=reward
            if done:
                break
        score_list.append(score)
        score_avg=np.mean(score_list[-100:])
        print(score_avg)
    # plt.plot([x for x in range(max_step)],score_list)
    # plt.xlabel('step')
    # plt.ylabel('average_score')
    # plt.show()




