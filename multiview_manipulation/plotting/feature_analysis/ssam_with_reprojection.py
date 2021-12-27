from datetime import datetime
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from manipulator_learning.sim.envs import *
from manipulator_learning.learning.agents.tf.common import CNNActor, convert_env_obs_to_tuple
import utils as tf_utils
from manipulator_learning.learning.utils.tf.general import set_training_seed

import utils as ssam_utils
import ellipsoid_tools as ell_tools
from multiview_manipulation import plotting as plot_utils

# CONFIG
# ----------------------------------------------------------------------------------------------
env_str = 'ThingDoorMultiview'
main_model_dir = '/home/trevor/data/paper-data/bc-view-agnostic/bc_models'
# model_name = 'ThingDoorMultiview_200_trajs_1'
model_name = 'ThingDoorImage_200_trajs_1'
model_dir = main_model_dir + '/' + model_name
main_exp_dir = '/media/trevor/Data/paper-data/bc-viewag'
run_dir = main_exp_dir + '/figures/feature_analysis/' + model_name + '_' + datetime.now().strftime("%y-%m-%d_%H-%M-%S")
show_opengl = False  # for actual videos/figures, this should be false, since the image rendering actually changes

gpu = 0  # policies that we're using won't load on a cpu b/c they use groups
env_seed = 0

# ssam stuff
ssam_img_freq = 1
num_eps = 5
num_highest_feats = 20
# feat_radius = .0025
feat_radius = .005

choose_feat_indices_once = True

# env segmentationMaskBuffer integers that we care about -- see pybullet docs on segmentationMaskBuffer for info

# DoorMultiview and DoorImage -- need to get these numbers if want to do for other environments
# robot segs correspond to main gripper, right finger, left finger, tool frame marker
# door segs correspond to door, handle front, handle top, handle bottom---specifically excluding frame
seg_mask_pairs = [
    [[1, 18], [1, 19], [1, 20], [1, 21]],  # robot
    # [[3, 1], [3, 2], [3, 3], [3, 4]]       # door -- excluding frame
    [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4]]       # door -- including frame
]
obj_to_cam_tfs = [
    [0, -.2, -.1, -np.pi/2, 0, np.pi],     # robot
    # [-.09, -.2, 0, -np.pi/2, 0, 0]         # door -- front view
    [-.2, -.1, .15, -3 * np.pi/4, 0, -np.pi/4]         # door -- iso view
]

# 1 step before initial grasp. could potentially also grab this automatically based on action
if 'Mult' in env_str:
    if 'Mult' in model_name:
        ep_render_starts = [16, 10, 22, 20, 14]
        # based on getting them from highest counts for all episodes, and then using for individual ep pb images
        # only has effect if generate_pb_plots_for_each_ep is True
        # feat_indices_for_objs = [[6, 8, 14], [17, 18, 19]]
        # left to right: 3rd highest to highest
        # 149 is blue top of gripper, 58 is light blue side, 99 is orange side
        # 81 is top of handle, 119 is dark green above handle, 158 is in between (light orange or light green)
        feat_indices_for_objs = [[149, 58, 99], [81, 119, 158]]  # need to use true (out of entire model) feat indices
        vid_feature_render_combos = [[99], [158], [149, 58, 99, 81, 119, 158]]
        vid_feature_render_cmap_inds = [[2], [5], [0, 1, 2, 3, 4, 5]]
        hardcode_feat_indices = np.array([102,  48, 133, 120, 151,  58, 149, 147, 141,  12, 124,  37,  19,
                                          104,  99,  81,  31,  63, 119, 158])
    elif 'Image' in model_name:
        ep_render_starts = [27, 8, 22, 25, 11]
        # feat_indices_for_objs = [[10, 16, 19], [2, 9, 18]]
        # feat_indices_for_objs = [[11, 23, 159], [25, 17, 6]]  # these might be better
        # 23 is top of gripper (blue), 110 is light blue near teeth, 159 left tooth/top (orange)
        # 25 is top of handle (light orange), 17 is bottom right of door (light green), 6 is across door (green)
        feat_indices_for_objs = [[23, 110, 159], [25, 17, 6]]
        vid_feature_render_combos = [[159], [6], [23, 110, 159, 25, 17, 6]]
        vid_feature_render_cmap_inds = [[2], [5], [0, 1, 2, 3, 4, 5]]
        hardcode_feat_indices = np.array([51, 157, 134, 123, 136, 122,   3,  83,  25,  20,  11,   5,  85,
                                          110, 114,  49,  23,  17,   6, 159])

num_timesteps_per_feat = 5
timestep_range_for_choosing_highest_feats = [ep_render_starts[0], ep_render_starts[0] + num_timesteps_per_feat]

OBJ_IDS = dict(robot=0, door=1)  # NOT the ids used in pybullet, but rather used in lists here
PB_OBJ_IDS = dict(robot=1, door=3)

obj_id_and_link_i = [3, 2]  # door object id, handle front link index

mpl_xlim = [0.55, 0.85]
mpl_ylim = [-.4, .05]
mpl_zlim = [.55, .9]

num_feats_per_obj = 3
num_feats_per_obj_all = (num_timesteps_per_feat - 1) * num_eps
render_individual_feats_in_mpl = False
generate_pb_plots_for_each_ep = True
generate_vid_images = True
# ----------------------------------------------------------------------------------------------

if generate_vid_images:
    feat_indices = hardcode_feat_indices

# get seg mask integers based on pybullet segmentationMaskBuffer --- value = objectUniqueId + (linkIndex+1)<<24
seg_mask_ints = []
for obj_i, obj in enumerate(seg_mask_pairs):
    seg_mask_ints.append([])
    for pair in obj:
        seg_mask_ints[obj_i].append(pair[0] + ((pair[1] + 1) << 24))

tf_utils.tf_setup(gpu)
set_training_seed(env_seed)  # not sure why, but if this isn't here, we get inconsistent runs

plot_utils.setup_pretty_plotting()

# env
action_mult = 1.0 if env_str != 'ThingLiftXYZStateMultiview' else .05
env = globals()[env_str](action_multiplier=action_mult, egl=not show_opengl, success_causes_done=True)
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

# store real world feature locations
all_feat_infos = []

cmap = plt.get_cmap("tab20")
feat_indices_chosen = False

env_cam = env.env.workspace_cam
pbc = env.env._pb_client

# convert xyz rpy transforms to xyz xyzw quat
obj_to_cam_pb_tfs = []
for obj_i, tf in enumerate(obj_to_cam_tfs):
    obj_to_cam_pb_tfs.append([*tf[:3], *pbc.getQuaternionFromEuler(tf[3:])])

for ep_i in range(num_eps):
    done = False
    ep_dir = run_dir + '/' + str(ep_i).zfill(4)
    os.makedirs(ep_dir)

    # need to record as lists since we don't immediately have the features we're going to use
    ep_obs = []
    ep_true_depths = []
    ep_seg_masks = []
    ep_highest_feats = []
    ep_ssam_out = []
    ep_robot_tool_cam_poses = []
    ep_obj_cam_poses = []
    ep_feat_infos = []

    obs = env.reset()

    # cam pose same for entire episode, so grab once
    world_to_cam_T = ssam_utils.get_world_to_cam_T_from_view_mat(env_cam._latest_view_mat)
    while not done:
        true_depth, seg_mask_img = ssam_utils.get_true_depth_and_segment_from_man_learn_env(env)
        act, var, ssam_out, raw_spatial_softmax = actor.inference(convert_env_obs_to_tuple(obs),
                                                                  also_output_ssam_values=True)
        act = act.numpy()

        max_spatial_softmax_activations = np.max(raw_spatial_softmax[0, :, :, :].numpy(), axis=(0, 1))

        # get positions of camera frames that are relative to gripper and object
        obj_cam_pose_pb = ssam_utils.get_link_to_pb_pose(pbc, obj_id_and_link_i[0], obj_id_and_link_i[1],
                                                 obj_to_cam_pb_tfs[1][:3], obj_to_cam_pb_tfs[1][3:])
        rob_tool_cam_pose_pb = ssam_utils.get_link_to_pb_pose(pbc, env.env.gripper.body_id,
                                                              env.env.gripper.manipulator._tool_link_ind,
                                                              obj_to_cam_pb_tfs[0][:3], obj_to_cam_pb_tfs[0][3:])

        # this way changes the feat indices every frame based on which are maximized
        if not feat_indices_chosen and not generate_vid_images:
            feat_indices = np.argsort(max_spatial_softmax_activations)[-num_highest_feats:]
            ep_highest_feats.append(feat_indices)

            # only need to save all of these since for the first ep, we don't know which features we're using
            ep_obs.append(obs)
            ep_true_depths.append(true_depth)
            ep_seg_masks.append(seg_mask_img)
            ep_ssam_out.append(ssam_out)
            ep_obj_cam_poses.append(obj_cam_pose_pb)
            ep_robot_tool_cam_poses.append(rob_tool_cam_pose_pb)

            # if choose_feat_indices_once and env.ep_timesteps >= num_timesteps_choose_highest_feats - 1:
            if choose_feat_indices_once and env.ep_timesteps >= timestep_range_for_choosing_highest_feats[1] - 1:
                ts_range = timestep_range_for_choosing_highest_feats
                ep_highest_feats = np.array(ep_highest_feats[ts_range[0]:ts_range[1] + 1]).flatten()
                unique, counts = np.unique(ep_highest_feats, return_counts=True)
                counts_sorted_i = np.argsort(counts)
                consistent_highest_feats = unique[counts_sorted_i[-num_highest_feats:]]
                feat_indices = consistent_highest_feats

                # need to go back to first set of observations and generate images from them, now that
                # features are chosen
                for ep_ts, (old_ep_obs, old_ssam_out, old_true_depth, old_seg_mask_img, old_obj_cam_pose_pb,
                            old_rob_tool_cam_pose_pb) in \
                        enumerate(zip(ep_obs, ep_ssam_out, ep_true_depths, ep_seg_masks, ep_obj_cam_poses,
                                      ep_robot_tool_cam_poses)):

                    # ssam_img = ssam_utils.convert_raw_ssam_to_img_coords(old_ssam_out, img_shape)

                    ssam_img = ssam_utils.get_ssam_on_img(old_ssam_out, feat_indices, old_ep_obs['img'], img_shape,
                                                          cmap, 5)

                    feat_infos = ssam_utils.get_feat_pos_and_obj(
                        pbc, feat_indices, ssam_img, env_cam, old_true_depth, world_to_cam_T, old_seg_mask_img,
                        seg_mask_ints, old_rob_tool_cam_pose_pb, old_obj_cam_pose_pb)

                    ep_feat_infos.append(feat_infos)

                    plt.savefig(ep_dir + '/' + str(ep_ts).zfill(4))



                # # render the points in a matplotlib 3d plot, where we can try creating convex hulls from them
                # fig = plt.figure()
                # ax = Axes3D(fig)
                #
                # all_feat_pos_world = ssam_utils.get_pos_worlds_from_ep_feat_infos(
                #     pbc, ep_feat_infos, [ep_render_starts[0], ep_render_starts[0] + num_timesteps_per_feat],
                #     rob_tool_cam_pose_pb, obj_cam_pose_pb)
                #
                # ssam_utils.single_ep_render_multiple_feats_in_mpl(ax, all_feat_pos_world, cmap, 'o')
                #
                # # render the points in pybullet and get an image
                # rgb, mbs = ssam_utils.single_ep_render_multiple_feats_in_pb(
                #     pbc, all_feat_pos_world, obj_cam_pose_pb, cmap, feat_radius)
                #
                # # ell_tools.plot_confidence_ellipsoid_from_points(ax, all_feat_pos_world[1, :, :-1].T, .95, cmap(1))
                #
                # # plt.imshow(rgb)
                # # plt.show()
                #
                # import ipdb; ipdb.set_trace()

                feat_indices_chosen = True

        # get ssam img
        if env.ep_timesteps % ssam_img_freq == 0 and \
                (not choose_feat_indices_once or (choose_feat_indices_once and feat_indices_chosen))\
                or generate_vid_images:

            ssam_img = ssam_utils.get_ssam_on_img(ssam_out, feat_indices, obs['img'], img_shape, cmap, 5)

            feat_infos = ssam_utils.get_feat_pos_and_obj(
                pbc, feat_indices, ssam_img, env_cam, true_depth, world_to_cam_T, seg_mask_img, seg_mask_ints,
                rob_tool_cam_pose_pb, obj_cam_pose_pb)
            ep_feat_infos.append(feat_infos)

            plt.savefig(ep_dir + '/' + str(env.ep_timesteps).zfill(4))

        # generate vid images at each comparison timestep and at each episode
        if generate_vid_images and \
                ep_render_starts[ep_i] + num_timesteps_per_feat > env.ep_timesteps >= ep_render_starts[ep_i]:

            all_feat_pos_world = np.array(ssam_utils.get_pos_worlds_from_ep_feat_infos(
                pbc, ep_feat_infos, [env.ep_timesteps, env.ep_timesteps + 1], rob_tool_cam_pose_pb,
                obj_cam_pose_pb))  # num feats, num ts, pts & obj

            pruned = []
            for i in [0, 1]:
                indices_for_objs_from_selected = []
                for index in feat_indices_for_objs[i]:
                    indices_for_objs_from_selected.append(np.argwhere(feat_indices == index)[0].item())
                pruned.append(all_feat_pos_world[indices_for_objs_from_selected])
            pruned = np.concatenate(pruned)

            vid_ep_dir = run_dir + '/vid_ep_' + str(ep_i).zfill(4)
            os.makedirs(vid_ep_dir, exist_ok=True)

            for vid_i, (feat_list, cmap_inds) in enumerate(zip(vid_feature_render_combos, vid_feature_render_cmap_inds)):
                feat_list_pruned = pruned[cmap_inds]
                rgb, mbs = ssam_utils.single_ep_render_multiple_feats_in_pb(
                    pbc, feat_list_pruned, obj_cam_pose_pb, cmap, feat_radius, make_objs_transparent=True, env=env,
                    use_robot_cam_pose=True, custom_cmap_inds=cmap_inds)
                im = Image.fromarray(rgb)
                feat_dir = vid_ep_dir + '/feat_comb_' + str(vid_i).zfill(4)
                os.makedirs(feat_dir, exist_ok=True)

                im.save(feat_dir + '/ts_%s.png' % str(env.ep_timesteps).zfill(4))

                # also test rendering in 2d
                twod_ssam_fig = plt.figure()
                ssam_utils.get_ssam_on_img(ssam_out, feat_list, obs['img'], img_shape, cmap, 100, False, cmap_inds)
                plt.axis('off')
                ax = twod_ssam_fig.gca()
                ax.set_axis_off()
                ax.autoscale(False)
                extent = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())

                # plt.savefig(feat_dir + '/mpl_ts_%s.png' % str(env.ep_timesteps).zfill(4), bbox_inches='tight')
                plt.savefig(feat_dir + '/mpl_ts_%s.png' % str(env.ep_timesteps).zfill(4), bbox_inches=extent)
                plt.close(twod_ssam_fig)


        # generate individual pb plots at each episode for comparisons
        if generate_pb_plots_for_each_ep and env.ep_timesteps == ep_render_starts[ep_i] + num_timesteps_per_feat:
            ts_range_list = [ep_render_starts[ep_i], ep_render_starts[ep_i] + num_timesteps_per_feat]
            all_feat_pos_world = np.array(ssam_utils.get_pos_worlds_from_ep_feat_infos(
                pbc, ep_feat_infos, ts_range_list, rob_tool_cam_pose_pb, obj_cam_pose_pb)) # num feats, num ts, pts & obj

            # before rendering pb image or doing anything else, switch to num_feats_per_obj features
            # need to use same features for each episode though, which we pick based on the final generated figure
            # and hardcode here
            pruned = []

            for i in [0, 1]:
                indices_for_objs_from_selected = []
                for index in feat_indices_for_objs[i]:
                    indices_for_objs_from_selected.append(np.argwhere(feat_indices == index)[0].item())
                pruned.append(all_feat_pos_world[indices_for_objs_from_selected])

            pruned = np.concatenate(pruned)

            rgb, mbs = ssam_utils.single_ep_render_multiple_feats_in_pb(
                pbc, pruned, obj_cam_pose_pb, cmap, feat_radius, make_objs_transparent=True, env=env,
                use_robot_cam_pose=True)

            im = Image.fromarray(rgb)
            im.save(run_dir + '/features_pb_ep_%s.png' % str(ep_i).zfill(4))


        # generate plots at this point, since this is when the objects are interacting
        if ep_i + 1 == num_eps and env.ep_timesteps == ep_render_starts[ep_i] + num_timesteps_per_feat:
            all_feat_infos.append(ep_feat_infos)

            fig = plt.figure(figsize=(6.4, 4.8))
            ax = Axes3D(fig)
            marker_list = ['$%d$' % i for i in range(num_eps)]
            all_eps_all_feat_pos_world = []
            # plot features for each episode
            for ep_plot_i, ep_feat_info_plot in enumerate(all_feat_infos):
                ts_range_list = [ep_render_starts[ep_plot_i], ep_render_starts[ep_plot_i] + num_timesteps_per_feat]
                # rgb, mbs = ssam_utils.single_ep_render_multiple_feats_in_pb(
                #     pbc, ep_feat_info_plot, ts_range_list, rob_tool_cam_pose_pb, obj_cam_pose_pb, cmap, feat_radius)

                all_feat_pos_world = ssam_utils.get_pos_worlds_from_ep_feat_infos(
                    pbc, ep_feat_info_plot, ts_range_list, rob_tool_cam_pose_pb, obj_cam_pose_pb)

                all_eps_all_feat_pos_world.append(all_feat_pos_world)

            # before rendering pb image or doing anything else, switch to num_feats_per_obj features
            all_eps_all_feat_pos_world = np.array(all_eps_all_feat_pos_world)  # num eps, num feats, num ts, pts & obj
            grouped_by_feat = np.concatenate(all_eps_all_feat_pos_world, axis=1)

            pruned_grouped_by_feat = []

            if generate_pb_plots_for_each_ep:  # meaning we're using hardcoded features defined above
                for i in [0, 1]:
                    indices_for_objs_from_selected = []
                    for index in feat_indices_for_objs[i]:
                        indices_for_objs_from_selected.append(np.argwhere(feat_indices == index)[0].item())
                    pruned_grouped_by_feat.append(grouped_by_feat[indices_for_objs_from_selected])

            else:  # meaning we're calculating most occurring features -- could (but shouldn't be) diff from hardcoded
                for i in [0, 1]:  # 0 for robot, 1 for gripper
                    counts = (grouped_by_feat[:, :, -1] == i).sum(axis=1)
                    obj_grouped_by_feat = grouped_by_feat[counts >= num_feats_per_obj_all][-num_feats_per_obj:]
                    if len(obj_grouped_by_feat) < num_feats_per_obj:
                        raise ValueError("Not enough consistently identified features based on "
                                         "num_feats_per_obj_all: %d and num_feats_per_obj: %d" %
                                         (num_feats_per_obj_all, num_feats_per_obj))
                    pruned_grouped_by_feat.append(obj_grouped_by_feat)
                    print("obj %d feats: %s" %
                          (i, feat_indices[np.argwhere(counts >= num_feats_per_obj_all)[-num_feats_per_obj:]]))

            new_grouped_by_feat = np.concatenate(pruned_grouped_by_feat)

            # render pb image, add scatter to mpl if desired

            # make robot and door partially transparent so all features show up
            alpha = 0.5
            color = .5
            env.env.update_body_visual(env.env.gripper.body_id, color, color, color, alpha)
            env.env.update_body_visual(env.env.door, color, color, color, alpha)

            all_mbs = []
            for ep_plot_i, ep_feat_info_plot in enumerate(all_feat_infos):
                ts_range_list = [ep_render_starts[ep_plot_i], ep_render_starts[ep_plot_i] + num_timesteps_per_feat]
                rgb, mbs = ssam_utils.single_ep_render_multiple_feats_in_pb(
                    pbc, new_grouped_by_feat, obj_cam_pose_pb, cmap, feat_radius)
                all_mbs.extend(mbs)

                if render_individual_feats_in_mpl:
                    all_feat_pos_world = ssam_utils.get_pos_worlds_from_ep_feat_infos(
                        pbc, ep_feat_info_plot, ts_range_list, rob_tool_cam_pose_pb, obj_cam_pose_pb)
                    ssam_utils.single_ep_render_multiple_feats_in_mpl(ax, all_feat_pos_world, cmap, marker_list[ep_plot_i])

            if generate_vid_images:
                for mb in all_mbs:
                    pbc.removeBody(mb)

                for vid_i, (feat_list, cmap_inds) in enumerate(
                        zip(vid_feature_render_combos, vid_feature_render_cmap_inds)):
                    all_mbs = []
                    feat_list_pruned = new_grouped_by_feat[cmap_inds]
                    for ep_plot_i, ep_feat_info_plot in enumerate(all_feat_infos):
                        rgb, mbs = ssam_utils.single_ep_render_multiple_feats_in_pb(
                            pbc, feat_list_pruned, obj_cam_pose_pb, cmap, feat_radius, make_objs_transparent=False,
                            env=env, custom_cmap_inds=cmap_inds)
                        all_mbs.extend(mbs)
                    im = Image.fromarray(rgb)
                    im.save(run_dir + '/all_eps_feat_comb_' + str(vid_i).zfill(4) + '.png')
                    for mb in all_mbs:
                        pbc.removeBody(mb)


            # render mpl image
            for feat_i, feat_points in enumerate(new_grouped_by_feat):
                # separate ellipses based on what obj they're on
                # now that we're using num_feats_per_obj, will just not render ellipses with few feats
                feat_points_rob = feat_points[feat_points[:, -1] == 0][:, :-1]
                feat_points_obj = feat_points[feat_points[:, -1] == 1][:, :-1]
                if len(feat_points_rob) > 10:
                    ell_tools.plot_confidence_ellipsoid_from_points(ax, feat_points_rob.T, 0.95, cmap(feat_i))
                if len(feat_points_obj) > 10:
                    ell_tools.plot_confidence_ellipsoid_from_points(ax, feat_points_obj.T, 0.95, cmap(feat_i))

                # feat_points_on_objects = feat_points[feat_points[:, -1] >= 0][:, :-1]
                # ell_tools.plot_confidence_ellipsoid_from_points(ax, feat_points_on_objects.T, 0.95, cmap(feat_i))

            if generate_vid_images:
                for vid_i, (feat_list, cmap_inds) in enumerate(
                        zip(vid_feature_render_combos, vid_feature_render_cmap_inds)):
                    vid_fig = plt.figure(figsize=(6.4, 4.8))
                    vid_ax = Axes3D(vid_fig)
                    for feat_i in cmap_inds:
                        feat_points = new_grouped_by_feat[feat_i]
                        feat_points_rob = feat_points[feat_points[:, -1] == 0][:, :-1]
                        feat_points_obj = feat_points[feat_points[:, -1] == 1][:, :-1]
                        if len(feat_points_rob) > 10:
                            ell_tools.plot_confidence_ellipsoid_from_points(vid_ax, feat_points_rob.T, 0.95, cmap(feat_i))
                        if len(feat_points_obj) > 10:
                            ell_tools.plot_confidence_ellipsoid_from_points(vid_ax, feat_points_obj.T, 0.95, cmap(feat_i))
                    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    vid_ax.set_xlim([mpl_xlim[0], mpl_xlim[1]])
                    vid_ax.set_ylim([mpl_ylim[0], mpl_ylim[1]])
                    vid_ax.set_zlim([mpl_zlim[0], mpl_zlim[1]])
                    vid_ax.set_xlabel("x (m)")
                    vid_ax.set_ylabel("y (m)")
                    vid_ax.set_zlabel("z (m)")
                    vid_ax.view_init(azim=120, elev=25)  # iso, matches pb
                    vid_fig.savefig(run_dir + '/vid_ell_comb_' + str(vid_i).zfill(4) + '.png',
                                bbox_inches=fig.bbox_inches.from_bounds(0.8, 0, 5.2, 4.4))


            # cur_base_pose = env.env.gripper.manipulator.get_link_pose(0)
            # cur_base_pose = [cur_base_pose[:3], cur_base_pose[3:]]
            # cur_base_pose_mat = trans_quat_to_mat(*cur_base_pose)
            # img, _ = env_cam.get_img(cur_base_pose_mat, width=1280, height=960)

            # 4 separate angles to show true ellipse sizes -- top, side, side, iso, plus pb image for comparison
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_xlim([mpl_xlim[0], mpl_xlim[1]])
            ax.set_ylim([mpl_ylim[0], mpl_ylim[1]])
            ax.set_zlim([mpl_zlim[0], mpl_zlim[1]])
            # ax.set_xticks(np.arange(mpl_xlim[0], mpl_xlim[1], .1))
            # ax.set_yticks(np.arange(mpl_ylim[0], mpl_ylim[1], .1))
            # ax.set_zticks(np.arange(mpl_zlim[0], mpl_zlim[1], .1))
            # plt.minorticks_on()

            # ax.xaxis.set_tick_params(labelsize=6)
            # ax.yaxis.set_tick_params(labelsize=6)
            # ax.zaxis.set_tick_params(labelsize=6)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")

            # ax.view_init(azim=115, elev=34)  # iso, matches pb
            ax.view_init(azim=120, elev=25)  # iso, matches pb
            # fig.savefig(run_dir + '/mpl_feats_iso.pdf', bbox_inches='tight')
            fig.savefig(run_dir + '/mpl_feats_iso.pdf', bbox_inches=fig.bbox_inches.from_bounds(0.8, 0, 5.2, 4.4))
            ax.view_init(azim=90, elev=0)  # front
            # fig.savefig(run_dir + '/mpl_feats_front.pdf', bbox_inches='tight')
            ax.set_yticks([])
            fig.savefig(run_dir + '/mpl_feats_front.pdf', bbox_inches=fig.bbox_inches.from_bounds(1.2, 0.8, 4.0, 3.0))
            ax.set_yticks(np.arange(mpl_ylim[0], mpl_ylim[1], .1))
            ax.view_init(azim=180, elev=0)  # right
            # fig.savefig(run_dir + '/mpl_feats_right.pdf', bbox_inches='tight')
            ax.set_xticks([])
            fig.savefig(run_dir + '/mpl_feats_right.pdf', bbox_inches=fig.bbox_inches.from_bounds(1.25, 0.8, 4.1, 3.075))
            ax.set_xticks(np.arange(mpl_xlim[0], mpl_xlim[1], .1))
            ax.view_init(azim=90, elev=90)  # top
            ax.set_zticks([])
            # fig.savefig(run_dir + '/mpl_feats_top.pdf', bbox_inches='tight')
            fig.savefig(run_dir + '/mpl_feats_top.pdf', bbox_inches=fig.bbox_inches.from_bounds(1.25, 0.6, 3.8, 3.8))
            ax.set_zticks(np.arange(mpl_zlim[0], mpl_zlim[1], .1))

            im = Image.fromarray(rgb)
            im.save(run_dir + '/features_pb.png')

            np.linspace(mpl_zlim[0], mpl_zlim[1], 3)

            ssam_fig = plt.figure()

        next_obs, rew, done, info = env.step(act)

        obs = next_obs

    all_feat_infos.append(np.array(ep_feat_infos))


# TODO next steps:
# 6. could also try to reproject 3d points back into 2d --- easy enough but try 3d first

# 7. (turns out probably don't need this)
#       NEED TO ADD SOMETHING TO IGNORE FEATURES THAT DON'T SHOW UP OFTEN ENOUGH ON EITHER OBJECT OR ROBOT