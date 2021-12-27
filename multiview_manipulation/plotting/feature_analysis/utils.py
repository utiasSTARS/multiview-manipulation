# Various utilities for SSAM feature analysis
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.interpolate import interp2d
import transforms3d as tf3d

from manipulator_learning.sim.utils.general import convert_quat_tf_to_pb, trans_quat_to_mat, convert_quat_pb_to_tf
from manipulator_learning.sim.robots.cameras import EyeInHandCam


# SSAM image tools
# --------------------------------------------------------------------------------------------------------------------

def convert_raw_ssam_to_img_coords(ssam, img_shape):
    """ Convert raw ssam with values in range  (-1, 1), output as (x0, y0, x1, y1, ..., xN, yN)
    to image coordinates for plotting over top of image.

    Also assumes that (-1, -1) corresponds to top left of image (0, 0). """
    ssam_x, ssam_y = (ssam[0][::2].numpy(), ssam[0][1::2].numpy())
    # ssam_x_img = (ssam_x * img_shape[1] / 2 + img_shape[1] / 2).astype(int)
    ssam_x_img = (ssam_x * img_shape[1] / 2 + img_shape[1] / 2)
    # ssam_y_img = (ssam_y * img_shape[0] / 2 + img_shape[0] / 2).astype(int)
    ssam_y_img = (ssam_y * img_shape[0] / 2 + img_shape[0] / 2)
    return ssam_x_img, ssam_y_img


def get_ssam_on_img(ssam, ssam_indices, img, img_shape, cmap, feat_size, include_labels=True, custom_cmap_inds=None):
    ssam_img = convert_raw_ssam_to_img_coords(ssam, img_shape)
    plt.cla()
    im = plt.imshow(img)
    # plt.scatter(ssam_img[0][:num_feats], ssam_img[1][:num_feats], c=np.arange(num_feats), s=5, cmap=cmap)
    # plt.scatter(ssam_img[0][ssam_indices], ssam_img[1][ssam_indices], c=ssam_indices, s=feat_size, cmap=cmap)

    if custom_cmap_inds is not None:
        colors = [cmap(i) for i in custom_cmap_inds]
        plt.scatter(ssam_img[0][ssam_indices], ssam_img[1][ssam_indices], c=colors, s=feat_size, edgecolors='k')
    else:
        plt.scatter(ssam_img[0][ssam_indices], ssam_img[1][ssam_indices], c=np.arange(len(ssam_indices)), s=feat_size,
                    cmap=cmap, edgecolors='k')

    if include_labels:
        labels = np.array(ssam_indices).astype(str)
        for feat_i, txt in zip(ssam_indices, labels):
            plt.annotate(txt, (ssam_img[0][feat_i] + .3, ssam_img[1][feat_i] + .3), fontsize=6)

    return ssam_img


# SE(3)/pybullet tools
# --------------------------------------------------------------------------------------------------------------------

def get_link_to_pb_pose(pb_client, body_id, link_i, trans, pb_rot):
    """ Get a pb T (7 tuple, 3 pos 4 xyzw quat) given a body id, link id, and a transformation """
    pbc = pb_client
    _, _, _, _, orig_pos, orig_rot = pbc.getLinkState(body_id, link_i)
    new_pos, new_rot = pbc.multiplyTransforms(orig_pos, orig_rot, trans, pb_rot)
    return (*new_pos, *new_rot)


def get_world_to_cam_T_from_view_mat(view_mat):
    """ Get the world to cam T matrix from the view matrix as output by pybullet """
    # view_mat from pybullet/opengl, after inversion, has z pointing inwards...180 rotation about x fixes that
    world_to_cam_T = np.linalg.inv(np.array(view_mat).reshape(4, 4).T)
    ogl_fix = tf3d.euler.euler2mat(np.pi, 0, 0)
    world_to_cam_T[:3, :3] = world_to_cam_T[:3, :3] @ ogl_fix

    return world_to_cam_T

# Matplotlib drawing tools
# --------------------------------------------------------------------------------------------------------------------


def single_ep_render_multiple_feats_in_mpl(axes_3d_obj, all_feat_pos_world, cmap, marker, plot_hull=False):
    # all_feat_pos_world = get_pos_worlds_from_ep_feat_infos(pb_client, ep_feat_infos, ts_range_list,
    #                                                        rob_tool_cam_pose_pb, obj_cam_pose_pb)

    ax = axes_3d_obj

    hulls = []
    hull_feat_is = []
    for feat_i, ep_feat_info in enumerate(all_feat_pos_world):
        if plot_hull:
            feat_on_objects_pos = ep_feat_info[ep_feat_info[:, -1] >= 0][:, :-1]
            if len(feat_on_objects_pos) >= 4:
                hull = ConvexHull(feat_on_objects_pos)
                hulls.append(hull)
                hull_feat_is.append(feat_i)

        for pos_obj_id in ep_feat_info:
            pos = pos_obj_id[:-1]
            o_id = int(pos_obj_id[-1])
            if o_id != -1:
                ax.scatter(*pos, color=cmap(feat_i), marker=marker)

    # add plotting of convex hulls
    if plot_hull:
        for hull, hull_feat_i in zip(hulls, hull_feat_is):
            for tri_i in hull.simplices:
                tri = Poly3DCollection(hull.points[tri_i])
                tri.set_color(cmap(hull_feat_i))
                # tri.set_edgecolor('k')  # adds extra edges that we might not want
                ax.add_collection3d(tri)


# Reprojection tools
# --------------------------------------------------------------------------------------------------------------------

def get_pos_worlds_from_ep_feat_infos(pb_client, ep_feat_infos, ts_range_list, rob_tool_cam_pose_pb, obj_cam_pose_pb):
    pbc = pb_client
    efi = np.array(ep_feat_infos)
    all_feat_world_pos = []

    for feat_loop_i in range(efi.shape[1]):
        feat_world_pos = []
        for pos_obj_id in efi[ts_range_list[0]:ts_range_list[1], feat_loop_i]:
            # get pos in world frame based on which obj it's in frame of
            pos = pos_obj_id[:-1]
            o_id = int(pos_obj_id[-1])
            cam_frames = [rob_tool_cam_pose_pb, obj_cam_pose_pb, [0, 0, 0, 0, 0, 0, 1]]
            pos_world, _ = pbc.multiplyTransforms(
                cam_frames[o_id][:3], cam_frames[o_id][3:], pos, [0, 0, 0, 1])
            feat_world_pos.append([*pos_world, o_id])
        all_feat_world_pos.append(feat_world_pos)
    return np.array(all_feat_world_pos)


def pb_take_cam_img(pb_client, env, obj_cam_pose_pb, use_robot_cam_pose=False):
    pbc = pb_client
    if use_robot_cam_pose:
        cur_base_pose = env.env.gripper.manipulator.get_link_pose(0)
        cur_base_pose = [cur_base_pose[:3], cur_base_pose[3:]]
        cur_base_pose_mat = trans_quat_to_mat(*cur_base_pose)
        rgb, _ = env.env.workspace_cam.get_img(cur_base_pose_mat, width=1280, height=960)
    else:
        cam = EyeInHandCam(pbc, [0, 0, 0], [0, 0, 0, 1], [0, 0, 1], [0, -1, 0], 'opengl', True,
                           width=1280, height=960)
        rgb, _ = cam.get_img(trans_quat_to_mat(obj_cam_pose_pb[:3], obj_cam_pose_pb[3:]))
    return rgb


def single_ep_render_multiple_feats_in_pb(pb_client, all_feat_pos_world, obj_cam_pose_pb, cmap, radius,
                                          make_objs_transparent=False, alpha=0.5, trans_color=0.5, env=None,
                                          use_robot_cam_pose=False, custom_cmap_inds=None):

    pbc = pb_client
    mbs = []
    for feat_i, ep_feat_info in enumerate(all_feat_pos_world):
        for pos_obj_id in ep_feat_info:
            pos = pos_obj_id[:-1]
            o_id = int(pos_obj_id[-1])
            if o_id != -1:
                if custom_cmap_inds is not None:
                    mb = draw_feature_in_pb(pbc, cmap(custom_cmap_inds[feat_i]), pos, radius=radius)
                else:
                    mb = draw_feature_in_pb(pbc, cmap(feat_i), pos, radius=radius)
                mbs.append(mb)

    if make_objs_transparent:
        color = trans_color

        # save all visual info about door and robot so we can reset it after making image
        door_ids = env.env.save_body_texture_ids(env.env.door)
        gripper_ids = env.env.save_body_texture_ids(env.env.gripper.body_id)

        env.env.update_body_visual(env.env.gripper.body_id, color, color, color, alpha)
        env.env.update_body_visual(env.env.door, color, color, color, alpha)

        rgb = pb_take_cam_img(pbc, env, obj_cam_pose_pb, use_robot_cam_pose)

        # remove the feature multibodies so they don't interfere with future episodes
        for mb in mbs:
            pbc.removeBody(mb)

        # reset visual info of door and robot
        env.env.update_body_visual_with_saved(env.env.door, door_ids, use_rgba=True)
        env.env.update_body_visual_with_saved(env.env.gripper.body_id, gripper_ids, use_rgba=True)
    else:
        rgb = pb_take_cam_img(pbc, env, obj_cam_pose_pb, use_robot_cam_pose)

    return rgb, mbs


def get_feat_pos_and_obj(pb_client, feat_indices, ssam_img, env_cam, true_depth, world_to_cam_T_mat, seg_mask_img,
                         seg_mask_ints, rob_tool_cam_pose_pb, obj_cam_pose_pb):
    """ Get feature infos: relative to cam pos and obj integer id for each feature. """
    pbc = pb_client
    feat_infos = []
    for feat_loop_i, feat in enumerate(feat_indices):
        u = ssam_img[0][feat]; v = ssam_img[1][feat]

        point_world = get_world_xyz_from_feature(
            u=u, v=v, fov=env_cam.fov, aspect=env_cam.aspect, depth_img=true_depth,
            world_to_cam_T=world_to_cam_T_mat)  # this is okay since it doesn't change for the whole episode

        # get object that feature point is "on"
        obj_id = -1  # -1 corresponds to not on any object of interest

        # avoid out of bound issues on too high u/v values
        width = seg_mask_img.shape[1]; height = seg_mask_img.shape[0]
        seg_mask_img_inds = [round(v), round(u)]
        if seg_mask_img_inds[0] == height:
            seg_mask_img_inds[0] -= 1
        if seg_mask_img_inds[1] == width:
            seg_mask_img_inds[1] -= 1

        feat_point_obj_id = seg_mask_img[seg_mask_img_inds[0], seg_mask_img_inds[1]]
        for int_list_i, int_list in enumerate(seg_mask_ints):
            if feat_point_obj_id in int_list:
                obj_id = int_list_i
                break

        # get pose of feature in "camera" frame for that object
        if obj_id > -1:
            cam_frames = [rob_tool_cam_pose_pb, obj_cam_pose_pb]
            cam_to_world_t, cam_to_world_r = pbc.invertTransform(cam_frames[obj_id][:3], cam_frames[obj_id][3:])
            feat_pos, feat_rot = pbc.multiplyTransforms(cam_to_world_t, cam_to_world_r, point_world, [0, 0, 0, 1])
        else:
            feat_pos = point_world

        # feat_info = dict(rel_pose=feat_pos, obj_id=obj_id)
        feat_info = [*feat_pos, obj_id]
        feat_infos.append(feat_info)
    return np.array(feat_infos)


def draw_feature_in_pb(pb_client, rgba_color, point_world, radius=.01, shape='SPHERE'):
    """ Draw a feature in pb. Return the pb body object id. """
    pbc = pb_client
    shape_options = ['SPHERE', 'BOX', 'CAPSULE', 'CYLINDER']
    assert shape in shape_options, "Shape %s is not an option out of options %s" % (shape, shape_options)
    shape = getattr(pbc, 'GEOM_' + shape)
    visual_shape_id = pbc.createVisualShape(shapeType=shape, rgbaColor=rgba_color, radius=radius)
    collision_shape_id = -1
    mb = pbc.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=collision_shape_id,
                             baseVisualShapeIndex=visual_shape_id,
                             basePosition=point_world,
                             useMaximalCoordinates=True)
    return mb


def get_true_depth_and_segment_from_man_learn_env(env):
    _, depth, segment = env.render('rgb_and_true_depth_and_segment_mask')
    return depth, segment


def get_world_xyz_from_feature(u, v, fov, aspect, depth_img, world_to_cam_T):
    """
    Get the xyz point corresponding to a 2D camera feature point, assuming that
    the distance from the camera eye is given by a depth image.

    The depth image from pybullet contains perpendicular distance from the plane of the camera, see
    https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer

    :param u:               x feature location in image, (0, 0) is top left
    :param v:               y feature location in image, (0, 0) is top left
    :param fov:             field of view as given to proj mat for pybullet camera, in degrees
    :param aspect:          aspect ratio as given to pybullet camera for generating image (often w/h)
    :param depth_img:       the full depth image, with TRUE depths. width and height taken from this.
    :param world_to_cam_T:  the 4x4 world to cam T, can be gotten using get_world_to_cam_T_from_view_mat
    :return:                [x, y, z] tuple of point in world frame
    """
    w = depth_img.shape[1]
    h = depth_img.shape[0]
    fov_rad = np.deg2rad(fov)
    depth_inter = interp2d(range(w), range(h), depth_img)

    d_uv = depth_inter(u, v)
    # d_uv = depth_img[v, u]  # first index is row which is Y value, second is col which is x value

    # if depth at any corner is significantly higher or lower than others (i.e. indicating depth is on an edge),
    # just take the depth at the rounded integer closest to the point, also ensuring it won't be out of bounds
    max_diff = .05  # 5cm
    if u < w - 1 and v < h - 1:   # this ensures avoidance of out of bounds
        depth_corners = depth_img[math.floor(v):math.ceil(v) + 1, math.floor(u):math.ceil(u) + 1]
        diffs = depth_corners - d_uv
        if np.any(diffs > max_diff):
            d_uv = depth_img[round(v), round(u)]
    elif u < w - 1 and v > h - 1:  # v is above max
        depth_corners = depth_img[h - 1, math.floor(u):math.ceil(u) + 1]  # only 2
        diffs = depth_corners - d_uv
        if np.any(diffs > max_diff):
            d_uv = depth_img[math.floor(v), round(u)]
    elif u > w - 1 and v < h - 1:  # u is above max
        depth_corners = depth_img[math.floor(v):math.ceil(v) + 1, w - 1]  # only 2
        diffs = depth_corners - d_uv
        if np.any(diffs > max_diff):
            d_uv = depth_img[round(v), math.floor(u)]

    # f_x = w / (2 * np.tan(fov_rad / 2))
    # turns out that FOV is actually vertical ---
    # see https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
    f_y = h / (2 * np.tan(fov_rad / 2))
    # f_x = f_y * aspect
    f_x = f_y
    theta_x = np.arctan((u - w/2) / f_x)
    theta_y = np.arctan((v - h/2) / f_y)
    point_homog = [d_uv * np.tan(theta_x), d_uv * np.tan(theta_y), d_uv, 1]
    point_world = world_to_cam_T.dot(point_homog)[:3]

    return point_world