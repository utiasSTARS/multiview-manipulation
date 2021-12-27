from datetime import datetime
import os
import matplotlib.pyplot as plt
from PIL import Image

from manipulator_learning.sim.envs import *
from manipulator_learning.learning.agents.tf.common import CNNActor, convert_env_obs_to_tuple
from multiview_manipulation import utils as tf_utils

# CONFIG
# ----------------------------------------------------------------------------------------------
env_str = 'ThingDoorMultiview'
main_model_dir = '/home/trevor/data/paper-data/bc-view-agnostic/bc_models'
fixed_model_name = 'ThingDoorImage_200_trajs_1'
mult_model_name = 'ThingDoorMultiview_200_trajs_1'
fixed_model_dir = main_model_dir + '/' + fixed_model_name
mult_model_dir = main_model_dir + '/' + mult_model_name
main_exp_dir = '/media/trevor/Data/paper-data/bc-viewag'
run_dir = main_exp_dir + '/figures/uncertainty_tests/' + env_str + '_' + datetime.now().strftime("%y-%m-%d_%H-%M-%S")

gpu = 0  # policies that we're using won't load on a cpu b/c they use groups
env_starting_seed = 0
num_eps = 20

# ----------------------------------------------------------------------------------------------

tf_utils.tf_setup(gpu)

# env
action_mult = 1.0 if env_str != 'ThingLiftXYZStateMultiview' else .05
env = globals()[env_str](action_multiplier=action_mult, egl=True, success_causes_done=True)

# actor
if env_str == "ThingLiftXYZStateMultiview":
    img_shape = (64, 48, 3)
    obs_shape = 9
else:
    img_shape = env.observation_space.spaces['img'].shape
    obs_shape = env.observation_space.spaces['obs'].shape[0]
act_shape = env.action_space.shape[0]
mult_actor = CNNActor(img_shape, obs_shape, act_shape, num_head_hidden=512, num_ensemble=5)
mult_actor.load_weights(mult_model_dir + '/bc_model')
fixed_actor = CNNActor(img_shape, obs_shape, act_shape, num_head_hidden=512, num_ensemble=5)
fixed_actor.load_weights(fixed_model_dir + '/bc_model')

cmap = plt.get_cmap("tab10")


def get_ep_uncertainties(env, actor):
    obs = env.reset()
    done = False
    ep_var = []
    ep_obs = []
    while not done:
        act, var = actor.inference(convert_env_obs_to_tuple(obs))
        act = act.numpy()
        ep_var.append(norm(var.numpy()))
        ep_obs.append(obs)

        # env.render()

        next_obs, rew, done, info = env.step(act)

        obs = next_obs

    return ep_var, ep_obs


def save_ep_imgs(dir, ep_obs):
    os.makedirs(dir, exist_ok=True)
    for ob_i, ob in enumerate(ep_obs):
        img = Image.fromarray(ob['img'])
        img.save(dir + '/' + str(ob_i).zfill(4) + '.png')


for ep_i in range(num_eps):
    uncertainty_fig = plt.figure()
    for actor, actor_str in zip([mult_actor, fixed_actor], ['mult_policy', 'fixed_policy']):
        env.seed(env_starting_seed + ep_i)
        ep_vars, ep_obs = get_ep_uncertainties(env, actor)
        save_ep_imgs(run_dir + '/' + actor_str + '_' + str(ep_i).zfill(4), ep_obs)
        plt.plot(ep_vars, label=actor_str)

    plt.legend()

    os.makedirs(run_dir, exist_ok=True)
    plt.savefig(run_dir + '/' + str(ep_i).zfill(4))
