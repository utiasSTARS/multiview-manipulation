from datetime import datetime
import os
import matplotlib.pyplot as plt

from manipulator_learning.sim.envs import *
from manipulator_learning.learning.agents.tf.common import CNNActor, convert_env_obs_to_tuple
import utils as tf_utils
import utils as ssam_utils


# CONFIG
# ----------------------------------------------------------------------------------------------
env_str = 'ThingDoorImage'
main_model_dir = '/home/trevor/data/paper-data/bc-view-agnostic/bc_models'
# model_name = 'ThingDoorImage_200_trajs_1'
model_name = 'ThingDoorImage_200_trajs_1'
model_dir = main_model_dir + '/' + model_name
main_exp_dir = '/media/trevor/Data/paper-data/bc-viewag'
run_dir = main_exp_dir + '/figures/feature_analysis/' + model_name + '_' + datetime.now().strftime("%y-%m-%d_%H-%M-%S")

gpu = 0  # policies that we're using won't load on a cpu b/c they use groups
env_seed = 0

# ssam stuff
ssam_img_freq = 1
num_eps = 5
num_highest_feats = 20
num_timesteps_choose_highest_feats = 30
choose_feat_indices_once = True
# ----------------------------------------------------------------------------------------------

tf_utils.tf_setup(gpu)

# env
action_mult = 1.0 if env_str != 'ThingLiftXYZStateMultiview' else .05
env = globals()[env_str](action_multiplier=action_mult, egl=True, success_causes_done=True)
env.seed(env_seed)

# actor
if env_str == "ThingLiftXYZStateMultiview":
    img_shape = (64, 48, 3)
    obs_shape = 9
else:
    img_shape = env.observation_space.spaces['img'].shape
    obs_shape = env.observation_space.spaces['obs'].shape[0]
act_shape = env.action_space.shape[0]
actor = CNNActor(img_shape, obs_shape, act_shape, num_head_hidden=512, num_ensemble=5)
actor.load_weights(model_dir + '/bc_model')

cmap = plt.get_cmap("tab10")
feat_indices_chosen = False


for ep_i in range(num_eps):
    done = False
    ep_dir = run_dir + '/' + str(ep_i).zfill(4)
    os.makedirs(ep_dir)
    ep_obs = []
    ep_highest_feats = []
    ep_ssam_out = []
    obs = env.reset()
    while not done:
        act, var, ssam_out, raw_spatial_softmax = actor.inference(convert_env_obs_to_tuple(obs),
                                                                  also_output_ssam_values=True)
        act = act.numpy()

        max_spatial_softmax_activations = np.max(raw_spatial_softmax[0, :, :, :].numpy(), axis=(0, 1))

        # this way changes the feat indices every frame based on which are maximized
        if not feat_indices_chosen:
            feat_indices = np.argsort(max_spatial_softmax_activations)[-num_highest_feats:]
            ep_obs.append(obs)
            ep_highest_feats.append(feat_indices)
            ep_ssam_out.append(ssam_out)
            if choose_feat_indices_once and env.ep_timesteps >= num_timesteps_choose_highest_feats - 1:
                ep_highest_feats = np.array(ep_highest_feats).flatten()
                unique, counts = np.unique(ep_highest_feats, return_counts=True)
                counts_sorted_i = np.argsort(counts)
                consistent_highest_feats = unique[counts_sorted_i[-num_highest_feats:]]
                feat_indices = consistent_highest_feats

                # need to go back to first set of observations and generate images from them, now that
                # features are chosen
                for ep_ts, (old_ep_obs, old_ssam_out) in enumerate(zip(ep_obs, ep_ssam_out)):
                    ssam_utils.get_ssam_on_img(old_ssam_out, feat_indices, old_ep_obs['img'], img_shape, cmap, 5)
                    plt.savefig(ep_dir + '/' + str(ep_ts).zfill(4))

                feat_indices_chosen = True

        # get ssam img
        if env.ep_timesteps % ssam_img_freq == 0:
            ssam_utils.get_ssam_on_img(ssam_out, feat_indices, obs['img'], img_shape, cmap, 5)
            plt.savefig(ep_dir + '/' + str(env.ep_timesteps).zfill(4))

        next_obs, rew, done, info = env.step(act)

        obs = next_obs