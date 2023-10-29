# import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/agent')
from lstm import LSTMModel



import utils
from encoder import Encoder
from rewarder import optimal_transport_plan, cosine_distance, euclidean_distance
import time
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR



class LSTMActor(nn.Module):
    def __init__(self, input_size, feature_dim, hidden_size, output_size):
        super(LSTMActor, self).__init__()
        self.lstm = LSTMModel(input_size, feature_dim, hidden_size, output_size)

    def get_feature(self, x):
        x_ = self.lstm.trunk(x)
        _, (out, _) = self.lstm.lstm(x_)
        out = out[-1:].permute(1, 0, 2).squeeze(dim=1)
        out = self.lstm.norm(out)
        return out
    
    def forward_with_hid(self, x, hidden=None):
        return self.lstm.forward_with_hid(x, hidden)


    def forward(self, obs, std):
        mu = self.lstm(obs)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist

class LSTMCritic(nn.Module):
    def __init__(self, input_size, feature_dim, hidden_size, action_size):
        super(LSTMCritic, self).__init__()
        self.trunk = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LeakyReLU(inplace=True))
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=feature_dim, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_size, hidden_size),
            nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True), nn.Linear(hidden_size, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_size, hidden_size),
            nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True), nn.Linear(hidden_size, 1))

    def forward(self, obs, action):
        h = self.trunk(obs)
        _, (h, _) = self.lstm(h)
        h = h[-1:].permute(1, 0, 2).squeeze(dim=1)
        h = self.relu(h)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2



class POTILAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, augment,
                 rewards, sinkhorn_rew_scale, update_target_every,
                 auto_rew_scale, auto_rew_scale_factor, obs_type, bc_weight_type, bc_weight_schedule):
        self.device = device
        self.lr = lr
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.augment = augment
        self.rewards = rewards
        self.sinkhorn_rew_scale = sinkhorn_rew_scale
        self.update_target_every = update_target_every
        self.auto_rew_scale = auto_rew_scale
        self.auto_rew_scale_factor = auto_rew_scale_factor
        self.use_encoder = True if obs_type == 'pixels' else False
        self.bc_weight_type = bc_weight_type
        self.bc_weight_schedule = bc_weight_schedule

        # models
        if self.use_encoder:
            self.encoder = Encoder(obs_shape).to(device)
            self.encoder_target = Encoder(obs_shape).to(device)
            repr_dim = self.encoder.repr_dim
        else:
            repr_dim = obs_shape[0]

        self.trunk_target = None # same actor

        self.actor = LSTMActor(repr_dim, feature_dim, hidden_dim, action_shape[0]).to(device)

        self.critic = LSTMCritic(repr_dim, feature_dim, hidden_dim, action_shape[0]).to(device)
        self.critic_target = LSTMCritic(repr_dim, feature_dim, hidden_dim, action_shape[0]).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = utils.RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def __repr__(self):
        return "potil"

    def train(self, training=True):
        self.training = training
        if self.use_encoder:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)

        obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)

        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, stddev)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        if self.use_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.use_encoder:
            self.encoder_opt.step()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        return metrics

    def update_actor(self, obs, obs_bc, obs_qfilter, action_bc, bc_regularize, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        # Compute bc weight
        if not bc_regularize:
            bc_weight = 0.0
        elif self.bc_weight_type == "linear":
            bc_weight = utils.schedule(self.bc_weight_schedule, step)
        elif self.bc_weight_type == "qfilter":
            """
            Soft Q-filtering inspired from 			
            Nair, Ashvin, et al. "Overcoming exploration in reinforcement 
            learning with demonstrations." 2018 IEEE international 
            conference on robotics and automation (ICRA). IEEE, 2018.
            """
            with torch.no_grad():
                stddev = 0.1
                dist_qf = self.actor_bc(obs_qfilter, stddev)
                action_qf = dist_qf.mean
                Q1_qf, Q2_qf = self.critic(obs_qfilter.clone(), action_qf)
                Q_qf = torch.min(Q1_qf, Q2_qf)
                bc_weight = (Q_qf > Q).float().mean().detach()

        actor_loss = - Q.mean() * (1 - bc_weight)

        if bc_regularize:
            stddev = 0.1
            dist_bc = self.actor(obs_bc, stddev)
            # print(action_bc)
            # print(dist_bc.log_prob(action_bc))
            log_prob_bc = dist_bc.log_prob(action_bc).sum(-1, keepdim=True)
            # 0.02 50
            actor_loss += - log_prob_bc.mean() * bc_weight * 2500

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor_q'] = Q.mean().item()
            if bc_regularize and self.bc_weight_type == "qfilter":
                metrics['actor_qf'] = Q_qf.mean().item()
            metrics['bc_weight'] = bc_weight
            metrics['regularized_rl_loss'] = -Q.mean().item() * (1 - bc_weight)
            metrics['rl_loss'] = -Q.mean().item()
            if bc_regularize:
                metrics['regularized_bc_loss'] = - log_prob_bc.mean().item() * bc_weight * 2500
                metrics['bc_loss'] = - log_prob_bc.mean().item() * 2500

        return metrics

    def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        if self.use_encoder and self.augment:
            obs_qfilter = self.aug(obs.clone().float())
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        else:
            obs_qfilter = obs.clone().float()
            obs = obs.float()
            next_obs = next_obs.float()

        if self.use_encoder:
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        if bc_regularize:
            batch = next(expert_replay_iter)
            obs_bc, action_bc = utils.to_torch(batch, self.device)
            # augment
            if self.use_encoder and self.augment:
                obs_bc = self.aug(obs_bc.float())
            else:
                obs_bc = obs_bc.float()
            # encode
            if bc_regularize and self.bc_weight_type == "qfilter":
                obs_qfilter = self.encoder_bc(obs_qfilter) if self.use_encoder else obs_qfilter
                obs_qfilter = obs_qfilter.detach()
            else:
                obs_qfilter = None
            obs_bc = self.encoder(obs_bc) if self.use_encoder else obs_bc
            # Detach grads
            obs_bc = obs_bc.detach()
        else:
            obs_qfilter = None
            obs_bc = None
            action_bc = None

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), obs_bc, obs_qfilter, action_bc, bc_regularize, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def ot_rewarder(self, observations, demos, step):

        if step % self.update_target_every == 0:  # todo: check not really equal
            if self.use_encoder:
                self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.trunk_target.load_state_dict(self.actor.trunk.state_dict())
            self.target_updated = True

        scores_list = list()
        ot_rewards_list = list()
        for demo in demos:
            obs = torch.tensor(observations).to(self.device).float()
            obs = self.trunk_target(self.encoder_target(obs)) if self.use_encoder else self.trunk_target(obs)
            exp = torch.tensor(demo).to(self.device).float()
            exp = self.trunk_target(self.encoder_target(exp)) if self.use_encoder else self.trunk_target(exp)
            obs = obs.detach()
            exp = exp.detach()

            if self.rewards == 'sinkhorn_cosine':
                cost_matrix = cosine_distance(
                    obs, exp)  # Get cost matrix for samples using critic network.
                transport_plan = optimal_transport_plan(
                    obs, exp, cost_matrix, method='sinkhorn',
                    niter=100).float()  # Getting optimal coupling
                ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
                    torch.mm(transport_plan,
                             cost_matrix.T)).detach().cpu().numpy()

            elif self.rewards == 'sinkhorn_euclidean':
                cost_matrix = euclidean_distance(
                    obs, exp)  # Get cost matrix for samples using critic network.
                transport_plan = optimal_transport_plan(
                    obs, exp, cost_matrix, method='sinkhorn',
                    niter=100).float()  # Getting optimal coupling
                ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
                    torch.mm(transport_plan,
                             cost_matrix.T)).detach().cpu().numpy()

            elif self.rewards == 'cosine':
                exp = torch.cat((exp, exp[-1].unsqueeze(0)))
                ot_rewards = -(1. - F.cosine_similarity(obs, exp))
                ot_rewards *= self.sinkhorn_rew_scale
                ot_rewards = ot_rewards.detach().cpu().numpy()

            elif self.rewards == 'euclidean':
                exp = torch.cat((exp, exp[-1].unsqueeze(0)))
                ot_rewards = -(obs - exp).norm(dim=1)
                ot_rewards *= self.sinkhorn_rew_scale
                ot_rewards = ot_rewards.detach().cpu().numpy()

            else:
                raise NotImplementedError()

            scores_list.append(np.sum(ot_rewards))
            ot_rewards_list.append(ot_rewards)

        closest_demo_index = np.argmax(scores_list)
        return ot_rewards_list[closest_demo_index]

    def save_snapshot(self):
        keys_to_save = ['actor', 'critic']
        if self.use_encoder:
            keys_to_save += ['encoder']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.use_encoder:
            self.encoder_target.load_state_dict(self.encoder.state_dict())
        # self.trunk_target.load_state_dict(self.actor.trunk.state_dict())

        if self.bc_weight_type == "qfilter":
            # Store a copy of the BC policy with frozen weights
            if self.use_encoder:
                self.encoder_bc = copy.deepcopy(self.encoder)
                for param in self.encoder_bc.parameters():
                    param.requires_grad = False
            self.actor_bc = copy.deepcopy(self.actor)
            for param in self.actor_bc.parameters():
                param.required_grad = False

        # Update optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    
    def train_bc_dist(self, obs_bc, action_bc):
        # global schedule
        # if schedule is False:
            # self.schedule = CosineAnnealingLR(self.actor_opt, T_max=50)
            # schedule = True

        metrics = dict()

        stddev = 0.01
        dist_bc = self.actor(obs_bc, stddev)
        log_prob_bc = dist_bc.log_prob(action_bc).sum(-1, keepdim=True)
        # 0.02 50
        actor_loss = -log_prob_bc.mean() * 0.0001

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        # self.schedule.step()
        
        metrics['bc_loss'] = actor_loss.item()

        return metrics
    
    def train_bc_norm(self, obs_bc, action_bc):
        metrics = dict()

        stddev = 0.01
        dist_bc = self.actor(obs_bc, stddev)
        loss = torch.abs(dist_bc.mean - action_bc).sum(-1)
        actor_loss = loss.mean() * 0.01

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        metrics['bc_loss'] = actor_loss.item()

        return metrics
    
    def train_bc_norm_step(self, obs_bc, action_bc):
        metrics = dict()

        out, hidden = self.actor.forward_with_hid(obs_bc)
        loss = torch.abs(out - action_bc).sum(-1)
        actor_loss = loss.mean() * 0.1

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        metrics['bc_loss'] = actor_loss.item()

        return metrics
    
    def act_with_hid(self, obs, hidden=None):
        obs = torch.as_tensor(obs, device=self.device)

        obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)

        action, hidden = self.actor.forward_with_hid(obs, hidden)
        return action.cpu().numpy()[0], hidden

schedule = False