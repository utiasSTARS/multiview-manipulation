# Test a set of BC policies from data based on desired number of expert demonstrations
import os
import pickle
import shutil

import tensorflow as tf

from manipulator_learning.learning.agents.tf.common import CNNActor
from multiview_manipulation.utils import eval_utils as eval_utils, tf_utils as tf_utils
from manipulator_learning.learning.imitation.intervenor import Intervenor
from manipulator_learning.learning.eval.data_recording import DataRecorder
from manipulator_learning.learning.utils.tf.rollouts import do_rollout
from manipulator_learning.learning.agents.tf.bc_policy import behavior_clone_save_load
from manipulator_learning.learning.data.tf.img_replay_buffer import ImgReplayBufferDisk
from manipulator_learning.sim.envs import *


# convenience function for multiple machines
# _, main_data_dir, bc_models_dir, expert_data_dir = eval_utils.default_arg_parser()

# Options ---------------------------------------------------------------------------------------------
main_data_dir = 'data'
bc_models_dir = 'bc_models'
expert_data_dir = 'demonstrations'
env_name = 'ThingDoorMultiview'
experiment_name = 'Interval25'
results_dir = main_data_dir + '/bc_results'
use_gpu = True
num_actor_ensemble = 5
num_actor_hidden = 512
bc_ckpts_num_traj = range(200, 24, -25)
starting_ep = 0  # in case we need to start from a particular episode
eps_per_policy = 50
device_type = 'vr'  # only used for real robot
bc_seeds = [1, 2, 3, 4, 5]
env_seed = 100
render = True
record_rb_first_seed_only = True
restart_current_num_dem_seed_batch = True  # discard data from a partially run set of eps_per_policy -- probably
# should only be used in sim
policy_env_name = 'ThingDoorMultiview'
device = 0  # which gpu

# Train options
also_train = True
expert_data_dir = main_data_dir + '/' + expert_data_dir + '/' + policy_env_name
batch_size = 64
loss_func = 'mse'
# -----------------------------------------------------------------------------------------------------

tf_utils.tf_setup(device)

# env setup
ros_env = False
if 'ThingRos' in env_name:
    from thing_gym_ros.envs import *

    ros_env = True
    env = globals()[env_name](reset_teleop_available=True, success_feedback_available=True)
    env.render()
    intervenor = Intervenor(device_type, env, real_time_multiplier=None)
elif 'Thing' in env_name or 'Panda' in env_name:  # sim env
    env = globals()[env_name]()
    intervenor = None
else:
    raise NotImplementedError("Not implemented for non manipulator-learning or thing-gym-ros envs yet")

obs_is_dict = type(env.observation_space) == gym.spaces.dict.Dict
if obs_is_dict:
    img_shape = env.observation_space.spaces['img'].shape
    obs_shape = env.observation_space.spaces['obs'].shape[0]
    act_shape = env.action_space.shape[0]
    actor = CNNActor(img_shape, obs_shape, act_shape, num_head_hidden=num_actor_hidden, num_ensemble=num_actor_ensemble)
else:
    raise NotImplementedError("Not implemented for non-dict observations yet")

main_results_dir = results_dir + '/' + env_name + '_' + str(env_seed) + '_' + experiment_name
if os.path.exists(main_results_dir + '/all_results.pkl'):
    main_recorder = DataRecorder.load_from_pickle(main_results_dir + '/all_results.pkl')
    completed_num_dem_seed_pairs = list(
        zip(main_recorder.per_ep_group_data['num_expert_demos'], main_recorder.per_ep_group_data['bc_seed']))
else:
    main_recorder = DataRecorder(main_results_dir,
                                 per_ep_group_keys=['bc_seed', 'num_expert_demos', 'avg_suc', 'avg_ret'])
    completed_num_dem_seed_pairs = []

for num_dem in bc_ckpts_num_traj:

    first_seed = True

    for s in bc_seeds:
        print("----------------------------------------------------------------------------------\n"
              "Starting testing for %d demos, %d policy seed" % (num_dem, s))

        cur_num_dem_seed_pair = tuple([num_dem, s])
        if cur_num_dem_seed_pair in completed_num_dem_seed_pairs:
            print("Data for num_dem seed pair already in main recorder, skipping!")
            continue

        writer = tf.summary.create_file_writer(main_results_dir + "/tb/%d_demos_%d_seed" % (num_dem, s))

        top_results_dir = main_results_dir + '/bc_seed' '_' + str(s)
        data_results_dir = top_results_dir + '/data_' + str(num_dem) + '_demos'
        rb_dir = top_results_dir + '/replay_buffer_' + str(num_dem) + '_demos'
        # there should be next to no data in the expert replay buffer, only there to work nicely with existing code in
        # do_rollout...
        erb_dir = top_results_dir + '/exp_replay_buffer_' + str(num_dem) + '_demos'

        if record_rb_first_seed_only and not first_seed:
            replay_buffer = None
            expert_replay_buffer = None
        else:
            replay_buffer = ImgReplayBufferDisk(rb_dir, obs_shape, act_shape)
            expert_replay_buffer = ImgReplayBufferDisk(erb_dir, obs_shape, act_shape)
        policy_total_numsteps = 0

        train_exp_rb = ImgReplayBufferDisk(expert_data_dir, obs_shape, act_shape) if also_train else None

        # new actor so we don't warm start the last policy, if we're training
        actor = CNNActor(img_shape, obs_shape, act_shape, num_head_hidden=num_actor_hidden,
                         num_ensemble=num_actor_ensemble)
        behavior_clone_save_load(actor, s, num_dem, policy_env_name, bc_models_dir, train_exp_rb, batch_size,
                                 writer=writer, loss_func=loss_func)

        # in case running policy stopped in the middle of testing, reload from where we were!
        if os.path.exists(data_results_dir + '/results.pkl') and not restart_current_num_dem_seed_batch:
            print("Data results pkl obj already exists -- loading existing data object, env.np_random state, and "
                  "replay buffer")
            recorder = DataRecorder.load_from_pickle(data_results_dir + '/results.pkl')
            env.np_random = pickle.load(open(data_results_dir + '/env_np_random.pkl', 'rb'))
        else:
            per_timestep_keys = ['reward', 'int_exp_in_control', 'in_exp_rb', 'rb_index']
            if os.path.exists(data_results_dir):
                shutil.rmtree(data_results_dir)
            if num_actor_ensemble > 1: per_timestep_keys.append('actor_variance')
            recorder = DataRecorder(data_results_dir,
                                    per_episode_keys=['success', 'ep_return'],
                                    per_timestep_keys=per_timestep_keys)
            env.seed(env_seed)

        for t in range(starting_ep, eps_per_policy):
            rew, timesteps, suc, auto_suc = do_rollout(env=env, actor=actor, replay_buffer=replay_buffer,
                                                       expert_replay_buffer=expert_replay_buffer, noise_scale=0.0,
                                                       intervenor=intervenor, always_allow_intervention=True,
                                                       data_recorder=recorder, render=render)
            print("Episode %d -- return: %.2f, success: %d" % (t, rew, suc))
            policy_total_numsteps += timesteps
            recorder.append_per_episode_data(dict(success=suc, ep_return=rew))
            recorder.internal_data['ep_num'] += 1
            recorder.internal_data['total_timesteps'] += timesteps
            recorder.save_all('results')
            pickle.dump(env.np_random, open(data_results_dir + '/env_np_random.pkl', 'wb'))
        avg_suc = np.array(recorder.per_episode_data['success']).mean()
        avg_ret = np.array(recorder.per_episode_data['ep_return']).mean()
        main_recorder.append_per_ep_group_data(
            dict(bc_seed=s, num_expert_demos=num_dem, avg_suc=avg_suc, avg_ret=avg_ret))
        main_recorder.save_all('all_results')

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('all eval/success rate', avg_suc, step=num_dem)
                tf.summary.scalar('all eval/average return', avg_ret, step=num_dem)

        starting_ep = 0  # so only applies to first policy that we want a new starting ep for

        first_seed = False
