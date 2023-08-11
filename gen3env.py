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
    def __init__(self) -> None:
        super().__init__()
        self.robot=Robot()
        action_low=[-1.,-1.,-1.,-1.]
        action_high=[1.,1.,1.,1.]
        obs_low=[-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]
        obs_high=[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
        self.action_space=spaces.Box(low=np.array(action_low),high=np.array(action_high),dtype=np.float64)#end_xyz+gripper
        self.observation_space=spaces.Box(low=np.array(obs_low),high=np.array(obs_high),dtype=np.float64)#end_xyz+gripper+peg_xyz+hole_xyz

    def reset(self):
        self.robot.init_scene()
        end_effect_pose=self.robot.get_cartesian_pose()
        end_x=end_effect_pose.position.x
        end_y=end_effect_pose.position.y
        end_z=end_effect_pose.position.z
        gripper_opening=self.robot.get_gripper_position()
        peg_x,peg_y,peg_z=self.robot.get_obj_pose('peg')
        hole_x,hole_y,hole_z=self.robot.get_obj_pose('hole')
        obs=np.array([end_x,end_y,end_z,gripper_opening,peg_x,peg_y,peg_z,hole_x,hole_y,hole_z])
        return obs 
    #get distance
    def compute_dis(self,obs):
        end_pose=obs[:4]
        peg_pose=obs[4:7]
        hole_pose=obs[7:]
        end_peg_dis=np.linalg.norm(end_pose-peg_pose)
        peg_hole_dis=np.linalg.norm(peg_pose-hole_pose)
        left_finger_pose=self.robot.get_link_pose('left_inner_finger')
        right_finger_pose=self.robot.get_link_pose('right_inner_finger')
        finger_pose=(left_finger_pose+right_finger_pose)/2
        grab_dis=np.linalg.norm(peg_pose,finger_pose)

        return end_peg_dis,peg_hole_dis,grab_dis

    def reach_dis(self,obs):
        end_peg_dis,_,_=self.compute_dis(obs)
        reach_reward=-end_peg_dis

        return reach_reward

    def insert_reward(self,obs):
        pass

    def grab_reward(self,obs):
        pass
    
    


    def compute_reward():
        pass

    def step(self,action):
        pass

        
        


        