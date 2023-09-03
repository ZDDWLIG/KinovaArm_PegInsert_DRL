# __init__.py
from gym.envs.registration import register

register(
    id='peg_in_hole-v0',
    entry_point='Gen3Env.gen3env:gen3env',
    max_episode_steps=50
)