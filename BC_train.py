import agent.potil_lstm as agent
from expert_dataloader import ExpertLoader, ExpertLoader_each_step
import agent.utils as utils
import sys
sys.path.append('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it')
import torch
torch.backends.cudnn.benchmark = True
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import queue


potil_agent = agent.POTILAgent(obs_shape=[4, 1], action_shape=[4, 1], device='cuda', lr=1e-3, feature_dim=512,
                               hidden_dim=128, critic_target_tau=0.01, num_expl_steps=0,
                               update_every_steps=2, stddev_schedule='linear(0.2,0.04,500000)', stddev_clip=0.1, use_tb=True, augment=True,
                               rewards='sinkhorn_cosine', sinkhorn_rew_scale=200, update_target_every=10000,
                               auto_rew_scale=True, auto_rew_scale_factor=10, obs_type='env', bc_weight_type='linear',
                               bc_weight_schedule='linear(0.15,0.03,20000)')


nstep = 32



# expert_loader = ExpertLoader(nstep, 50)
# expert_iter = iter(expert_loader)
# test_data = expert_loader.read_data(0,1)
# state, action = test_data
# seq_queue = queue.Queue(nstep)
# test_data_storage = []
# for i in range(nstep):
#     seq_queue.put(state[0])
# for i in range(state.shape[0]):
#     seq_queue.get()
#     seq_queue.put(state[i])
#     test_data_storage.append((np.stack(seq_queue.queue), action[i]))

# with step
expert_loader = ExpertLoader_each_step(nstep, 50)
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


cnt = 1
# def test():
#     obs = utils.to_torch(test_data, 'cuda')[0]
#     replay = []
#     for i in range(80):
#         with torch.no_grad():
#             action = potil_agent.act(obs, 0, True)
#         action[0] = (action[0] + 1.3) * 0.2 + 0.2
#         action[1] = (action[1] + 1) * 1.1 * -0.1
#         action[2] = (action[2] + 0.75) * 0.125 + 0.05
#         action[3] = (action[3] + 1) * 0.2
#         obs[:4] = torch.from_numpy(action).cuda()
#         replay.append(obs)

#     fig = plt.figure()
#     #创建3d绘图区域
#     ax = plt.axes(projection='3d')
#     #从三个维度构建
#     z = test_data[0][:,2]
#     x = test_data[0][:,0]
#     y = test_data[0][:,1]
#     #调用 ax.plot3D创建三维线图
#     ax.scatter3D(x, y, z,'gray', s=1)
#     replay = np.stack(replay)
#     z = replay[:,2]
#     x = replay[:,0]
#     y = replay[:,1]
#     #调用 ax.plot3D创建三维线图
#     ax.scatter3D(x, y, z,'gray', s=1)
#     plt.savefig(f'./figs/{cnt}.png')
#     cnt += 1

def scale(action_):
    action = action_.copy()
    action[0] = (action[0] + 1.31) * 0.2 + 0.2
    action[1] = (action[1] + 0.84) * 1.1 * -0.11
    action[2] = (action[2] + 0.83) * 0.15 + 0.05
    action[3] = (action[3] + 1) * 0.25
    return action



def test():
    loss = 0
    for data in test_data_storage:
        data = utils.to_torch(data, 'cuda')
        with torch.no_grad():
            action = potil_agent.act(data[0][:, :4], 0, True)
        loss += np.linalg.norm(scale(action) - scale(data[1].cpu().numpy()), ord=1)
    loss /= state.shape[0]
    print(loss)


# for i in range(100000):
#     batch = next(expert_iter)
#     obs_bc, action_bc = utils.to_torch(batch, 'cuda')
#     metrics = potil_agent.train_bc_norm(obs_bc, action_bc)
#     # if i % 10 == 1:
#         # print(metrics['bc_loss'])
#     if i % 20 == 0:
#         test()
#     if i % 100 == 0:
#         torch.save(potil_agent.actor, 'BC_actor_norm.pkl')

# for i in range(100000):
#     batch = next(expert_iter)
#     obs_bc, action_bc = utils.to_torch(batch, 'cuda')
#     metrics = potil_agent.train_bc_dist(obs_bc, action_bc)
#     # if i % 10 == 1:
#         # print(metrics['bc_loss'])
#     if i % 100 == 0:
#         test()
#     if i % 100 == 0:
#         torch.save(potil_agent.actor, 'BC_actor_dist.pkl')


# potil_agent.actor = torch.load('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/BC_actor_dist.pkl').cuda()
potil_agent.actor.train()
potil_agent.actor_opt = torch.optim.Adam(potil_agent.actor.parameters(), lr=potil_agent.lr)
for i in range(100000):
    batch = next(expert_iter)
    obs_bc, action_bc = utils.to_torch(batch, 'cuda')
    metrics = potil_agent.train_bc_norm_step(obs_bc[:,:,:4], action_bc)
    # if i % 10 == 1:
        # print(metrics['bc_loss'])
    if i % 100 == 0:
        test()
    if i % 100 == 0:
        torch.save(potil_agent.actor, 'BC_actor_dist.pkl')






