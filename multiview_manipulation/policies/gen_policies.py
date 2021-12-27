# Generate a set of BC policies from data based on desired number of expert demonstrations
import copy

from manipulator_learning.learning.agents.tf.common import Actor, CNNActor
import multiview_manipulation.utils.eval_utils as eval_utils
import multiview_manipulation.utils.tf_utils as tf_utils
from manipulator_learning.learning.data.tf.img_replay_buffer import ImgReplayBufferDisk
from manipulator_learning.learning.agents.tf.bc_policy import behavior_clone_save_load


# convenience function for multiple machines
# _, _, bc_models_dir, expert_data_dir = eval_utils.default_arg_parser()

# Options ---------------------------------------------------------------------------------------------
bc_models_dir = 'data/bc_models'
expert_data_dir = 'data/demonstrations'
dataset_dir = 'ThingDoorMultiview'
env_name = 'ThingDoorMultiview'
expert_data_dir = expert_data_dir + '/' + dataset_dir
use_gpu = True
num_actor_ensemble = 5
num_actor_hidden = 512
batch_size = 64
bc_ckpts_num_traj = range(25, 201, 25)
obs_is_dict = True
seeds = [1, 2, 3, 4, 5]
device = 0  # which gpu
# -----------------------------------------------------------------------------------------------------

tf_utils.tf_setup(device)  # limit gpu memory growth, set device

if obs_is_dict:
    bc_rb = ImgReplayBufferDisk(expert_data_dir)
    act_shape = bc_rb.act_dim
    obs_shape = bc_rb.state_dim
    img_shape = tuple(
        bc_rb.dataset.load_to_ram_worker(0)[0].shape)
    bc_rb.dataset.load_to_ram(8, True, True, True)  # crucial!! otherwise tensorflow and multiprocessing don't play nice
else:
    raise NotImplementedError("Implement when needed")
    act_shape = env.action_space.shape[0]
    obs_shape = env.observation_space.shape[0]
    img_shape = None

for num_dems in bc_ckpts_num_traj:
    for s in seeds:
        if obs_is_dict:
            num_demos_rb = bc_rb.get_copy(use_same_dataset_obj=True)
            actor = CNNActor(img_shape, obs_shape, act_shape, num_head_hidden=num_actor_hidden,
                             num_ensemble=num_actor_ensemble)
        else:
            num_demos_rb = copy.deepcopy(bc_rb)
            actor = Actor(obs_shape, act_shape, num_head_hidden=num_actor_hidden, num_ensemble=num_actor_ensemble)
        if not obs_is_dict:
            obs, actions, _, _, _, _ = num_demos_rb.get_data_numpy()

        print("Starting training for %d demos from %s dataset" % (num_dems, dataset_dir))

        behavior_clone_save_load(actor, s, num_dems, env_name, bc_models_dir, num_demos_rb, batch_size)
