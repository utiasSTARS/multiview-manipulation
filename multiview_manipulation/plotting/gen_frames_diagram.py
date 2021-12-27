import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import time

from manipulator_learning.sim.envs import *
import manipulator_learning.learning.data.img_depth_dataset as img_depth_dataset
from manipulator_learning.learning.imitation.device_utils import CollectDevice
from manipulator_learning.sim.envs.pb_env import add_pb_frame_marker, add_pb_frame_marker_by_pose

# note: to get the kinect to show up, uncomment out the part in color blocks under "for generating pretty images"

# CONFIG
#--------------------------------------------------------------------------------
env_str = 'ThingDoorMultiview'
num_episodes = 10

seed = 0
main_exp_dir = '/media/trevor/Data/paper-data/bc-viewag'
exp_dir = main_exp_dir + '/figures/frames/' + datetime.now().strftime("%y-%m-%d_%H-%M-%S")
# debug_camera_angle = [[.36, -.19, .62], 1.2, -38.6, -548.4]  # side view
debug_camera_angle = [[.24, -.13, .26], 1.6, -65.4, -704]  # top view from opposite of cam
# ex_base_angles = np.linspace(-3 * np.pi / 16, np.pi / 16, num=4)
# ex_base_angles = np.linspace(-3 * np.pi / 16, np.pi / 16, num=3)  # used for paper
ex_base_angles = np.linspace(-3 * np.pi / 16, np.pi / 16, num=6)  # used for video
frames_labelled_base_angle = -2.5 * np.pi / 16

#--------------------------------------------------------------------------------

# angles used for door -- random_base_theta_bounds = (-3 * np.pi / 16, np.pi / 16)

# env = globals()[env_str](action_multiplier=1.0, egl=False, robot='thing_panda_gripper_fake_cam_mount')
env = globals()[env_str](action_multiplier=1.0, egl=False, robot='thing_panda_gripper_no_collision')
env.seed(seed)
dev = CollectDevice('keyboard', env.env.gripper.valid_t_dof, env.env.gripper.valid_r_dof, env.grip_in_action, .3)
os.makedirs(exp_dir)
pbc = env.env._pb_client

# setting debug camera
pbc.resetDebugVisualizerCamera(cameraTargetPosition=debug_camera_angle[0],
                cameraDistance=debug_camera_angle[1], cameraPitch=debug_camera_angle[2],
                cameraYaw=debug_camera_angle[3])

img_traj_data = []
depth_traj_data = []

act_dim = env.action_space.shape[0]
obs_shape = env.observation_space.spaces['obs'].shape[0]
ds = img_depth_dataset.Dataset(exp_dir, state_dim=obs_shape, act_dim=act_dim)


for ep_i in range(num_episodes):
    obs = env.reset(mb_base_angle=ex_base_angles[ep_i])  # for generating sepcific angles fig
    # obs = env.reset(mb_base_angle=frames_labelled_base_angle)  # for generating frames fig
    done = False
    while not done:
        act = np.zeros(act_dim)
        act[-1] = -1
        next_obs, rew, done, info = env.step(act)
        env.ep_timesteps = 0

        rk = ord('r')
        keys = pbc.getKeyboardEvents()
        if rk in keys and keys[rk] and pbc.KEY_WAS_TRIGGERED:
            break

        # img, depth = env.env.render("robot_side_cam")


