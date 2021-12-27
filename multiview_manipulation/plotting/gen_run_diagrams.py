from datetime import datetime

from manipulator_learning.sim.envs import *
import manipulator_learning.learning.data.img_depth_dataset as img_depth_dataset
from manipulator_learning.learning.imitation.device_utils import CollectDevice
from manipulator_learning.learning.agents.tf.bc_policy import behavior_clone_save_load
from manipulator_learning.learning.agents.tf.common import CNNActor
from manipulator_learning.learning.agents.tf.common import convert_env_obs_to_tuple
import multiview_manipulation.utils.tf_utils as tf_utils
from manipulator_learning.learning.utils.tf.general import set_training_seed

# CONFIG
#--------------------------------------------------------------------------------
env_str = 'ThingDoorMultiview'
num_suc_episodes = 10
seed = 0
device = 0
starting_ep = 0
render = True

# policy options
main_policy_dir = '/home/trevor/data/paper-data/bc-view-agnostic/bc_models'
num_trajs = 200
policy_seed = 1
policy_env_str = "ThingDoorMultiview"


main_exp_dir = '/media/trevor/Data/paper-data/bc-viewag'
run_dir = main_exp_dir + '/figures/runs/' + env_str + '_' + str(num_trajs) + '_' + str(policy_seed) + \
          '_' + str(num_suc_episodes) + '_eps_' + datetime.now().strftime("%y-%m-%d_%H-%M-%S")
# debug_camera_angle = [[.36, -.19, .62], 1.2, -38.6, -548.4]  # side view
debug_camera_angle = [[.24, -.13, .26], 1.6, -65.4, -704]  # top view from opposite of cam
# ex_base_angles = np.linspace(-3 * np.pi / 16, np.pi / 16, num=4)
ex_base_angles = np.linspace(-3 * np.pi / 16, np.pi / 16, num=3)
frames_labelled_base_angle = -2.5 * np.pi / 16

#--------------------------------------------------------------------------------

tf_utils.tf_setup(device)

# angles used for door -- random_base_theta_bounds = (-3 * np.pi / 16, np.pi / 16)

# env = globals()[env_str](action_multiplier=1.0, egl=False, robot='thing_panda_gripper_fake_cam_mount')
if env_str == 'ThingLiftXYZStateMultiview':
    action_mult = .05
    extra_args = dict(success_on_done=True)
else:
    action_mult = 1.0
env = globals()[env_str](action_multiplier=action_mult, egl=True, success_causes_done=True)
env.seed(seed)
set_training_seed(100)  # not sure why, but if this isn't here, we get inconsistent runs

if env_str == "ThingLiftXYZStateMultiview":
    img_shape = (64, 48, 3)
    obs_shape = 9
else:
    img_shape = env.observation_space.spaces['img'].shape
    obs_shape = env.observation_space.spaces['obs'].shape[0]
act_shape = env.action_space.shape[0]
actor = CNNActor(img_shape, obs_shape, act_shape, num_head_hidden=512, num_ensemble=5)
behavior_clone_save_load(actor, policy_seed, num_trajs, policy_env_str, main_policy_dir, expert_replay_buffer=None)

dev = CollectDevice('keyboard', env.env.gripper.valid_t_dof, env.env.gripper.valid_r_dof, env.grip_in_action, .3)
pbc = env.env._pb_client

# setting debug camera
pbc.resetDebugVisualizerCamera(cameraTargetPosition=debug_camera_angle[0],
                cameraDistance=debug_camera_angle[1], cameraPitch=debug_camera_angle[2],
                cameraYaw=debug_camera_angle[3])

img_traj_data = []
depth_traj_data = []

ds_hq = img_depth_dataset.Dataset(run_dir + '/hq', state_dim=obs_shape, act_dim=act_shape)
ds = img_depth_dataset.Dataset(run_dir + '/cam_view', state_dim=obs_shape, act_dim=act_shape)

suc_ep_count = 0
ep_i = 0
while suc_ep_count < num_suc_episodes:
    if ep_i < starting_ep:
        ep_i += 1
        continue
    obs = env.reset()
    if env_str == "ThingLiftXYZStateMultiview":
        obs = env.get_img_obs()
    img, depth = env.env.render("robot_side_cam")
    # obs = env.reset(mb_base_angle=frames_labelled_base_angle)
    done = False
    hq_traj_imgs = [img]
    # hq_traj_depths = [depth]
    hq_traj_depths = [np.array([0])]  # makes folders smaller
    traj_imgs = [obs['img']]
    traj_depths = [np.array([0])]
    traj_states = [np.concatenate([obs['obs'], np.zeros(act_shape).flatten(), np.array([0]),
                                           np.array([0]), np.array([False])])]
    while not done:
        act, var = actor.inference(convert_env_obs_to_tuple(obs))
        act = act.numpy()
        next_obs, rew, done, info = env.step(act)
        if env_str == "ThingLiftXYZStateMultiview":
            next_obs = env.get_img_obs()

        # rk = ord('r')
        # keys = pbc.getKeyboardEvents()
        # if rk in keys and keys[rk] and pbc.KEY_WAS_TRIGGERED:
        #     break


        img, depth = env.env.render("robot_side_cam")
        hq_traj_imgs.append(img)
        hq_traj_depths.append(np.array([0]))
        traj_states.append(np.concatenate([obs['obs'], np.array(act).flatten(), np.array([rew]),
                                           np.array([0]), np.array([done])]))

        obs = next_obs
        traj_imgs.append(obs['img'])
        traj_depths.append(np.array([0]))

        if render:
            env.render()

    ep_i += 1
    if info['done_success']:
        suc_ep_count += 1
        ds_hq.append_traj_data_lists(traj_states, hq_traj_imgs, hq_traj_depths, final_obs_included=True)
        ds.append_traj_data_lists(traj_states, traj_imgs, traj_depths, final_obs_included=True)
