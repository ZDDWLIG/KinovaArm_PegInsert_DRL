from typing import Optional, Tuple
import gym 
# import numpy as np
# from stable_baselines3 import PPO
import sys
import time
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import gym.spaces as spaces
from robot import Robot
# TASK_LIST={
#     'peg_in_hole':
# }

class gen3env(gym.Env):
    metadata={'render.modes':['human','rgb_array']}
    def __init__(self):
        super(gen3env,self).__init__()
        self.robot=Robot()
        self.peg_pose=[0.1,0,0.3]
        self.hole_pose=[1.,-0.2,0.3]
        self.info=None
        action_low=[-1.,-1.,-1.,-1.]
        action_high=[1.,1.,1.,1.]
        obs_low=[-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]
        obs_high=[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
        self.action_space=spaces.Box(low=np.array(action_low),high=np.array(action_high),dtype=np.float64)#end_xyz+gripper
        self.observation_space=spaces.Box(low=np.array(obs_low),high=np.array(obs_high),dtype=np.float64)#end_xyz+gripper+peg_xyz+hole_xyz

    def reset(self):
        
        self.robot.remove_scene()
        self.robot.init_scene(peg_pose=self.peg_pose,hole_pose=self.hole_pose)
        obs=self.get_obs()
        self.render()

        return obs
    #get distance
    def compute_dis(self,obs):
        end_pose=obs[:3]
        peg_pose=obs[4:7]
        # print('peg',peg_pose,type(peg_pose))
        hole_pose=obs[7:]
        end_peg_dis=np.linalg.norm(end_pose-peg_pose)
        peg_hole_dis=np.linalg.norm(peg_pose-hole_pose)
        left_finger_pose=self.robot.get_link_pose('left_inner_finger')
        right_finger_pose=self.robot.get_link_pose('right_inner_finger')
        finger_pose=(left_finger_pose+right_finger_pose)/2       
        # print('finger',finger_pose,type(finger_pose))
        grab_dis=np.linalg.norm(peg_pose-finger_pose)

        return end_peg_dis,peg_hole_dis,grab_dis
        
    def do_sim(self,action):
        action,_=self.scale(action=action)
        self.robot.move(pose=action[:-1])
        self.robot.reach_gripper_position(action[-1])

    def get_obs(self):
        end_effect_pose=self.robot.get_cartesian_pose()
        end_x=end_effect_pose.position.x
        end_y=end_effect_pose.position.y
        end_z=end_effect_pose.position.z
        gripper_opening=self.robot.get_gripper_position()
        peg_x,peg_y,peg_z=self.robot.get_obj_pose('peg')
        hole_x,hole_y,hole_z=self.robot.get_obj_pose('hole')
        obs=np.array([end_x,end_y,end_z,gripper_opening,peg_x,peg_y,peg_z,hole_x,hole_y,hole_z])
        return obs 
    
    def compute_reward(self,action,obs):
        end_peg_dis,peg_hole_dis,grab_dis=self.compute_dis(obs)
        peg_pose=obs[4:7]
        left_finger_pose=self.robot.get_link_pose('left_inner_finger')
        right_finger_pose=self.robot.get_link_pose('right_inner_finger')
        finger_pose=(left_finger_pose+right_finger_pose)/2
        
        def is_pick():
            return (finger_pose[2]-peg_pose[2])<0.01
        self.is_pick=is_pick()
        def is_drop():
            return ((peg_hole_dis>0.05)and (grab_dis>0.05))
        
        def reach_reward():
            reach_reward=-end_peg_dis
            if grab_dis<0.05:
                reach_reward+=min(action[3],1)/50
            
            return reach_reward,grab_dis
        
        def pick_reward():
            if self.is_pick and not (is_drop()):
                return 10
            else:
                return 0

        def insert_reward():
            # c1 = 1000
            # c2 = 0.01
            # c3 = 0.001
            cond=self.is_pick and not(is_drop())and (grab_dis<0.05)
            if cond :
                insert_reward=-peg_hole_dis
            else:
                insert_reward=-10

            return [insert_reward,peg_hole_dis]
        reach_rew,grab_dis=reach_reward()
        pick_rew=pick_reward()
        insert_rew,peg_hole_dis=insert_reward()
        reward=reach_rew+pick_rew+insert_rew
        return[reward,reach_rew,pick_rew,insert_rew,peg_hole_dis]        


        

    def step(self,action):
        self.do_sim(action)
        obs=self.get_obs()
        # print('='*50)
        # print('before scale',obs)
        reward,reach_rew,pick_rew,insert_rew,peg_hole_dis=self.compute_reward(action,obs)
        info={
            "reward":reward,
            "reach_rew":reach_rew,
            "pick_rew":pick_rew,
            "insert_rew":insert_rew,
            "peg_hole_dis":peg_hole_dis
        }
        done=bool(peg_hole_dis<0.005)
        _,obs=self.scale(obs=obs)
        # print('='*50)
        # print('after scale',obs)
        self.info=info
        self.render()
        return obs,reward,done,info
    def render(self, mode='human'):
        print(self.info)
    
    def close(self):
        pass

    def scale(self,action=None,obs=None):
        if action is not None:
            action[0]=(action[0]+1)*0.25+0.5
            action[1]=(action[1])*0.5
            action[2]=(action[2]+1)*0.25
            action[3]=(action[3]+1)*0.25

        if obs is not None:
            obs[0]=(obs[0]+1)*0.25+0.5
            obs[1]=(obs[1])*0.5
            obs[2]=(obs[2]+1)*0.25
            obs[3]=(obs[3]+1)*0.25
            obs[4]=(obs[0]+1)*0.25+0.5
            obs[5]=(obs[1])*0.5
            obs[6]=(obs[2]+1)*0.25
            obs[7]=(obs[0]+1)*0.25+0.5
            obs[8]=(obs[1])*0.5
            obs[9]=(obs[2]+1)*0.25
        return action,obs
        
        


        
