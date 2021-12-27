import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from multiview_manipulation import plotting as plot_utils

# CONFIG
#-------------------------------------------------------------------------------
# plot options
plt_shape = [2, 2]
font_size = 22
interval_percentile = 95
plot_width = 3.2
plot_height = 2.4
dist_line_width = 3
dist_line_color = 'red'
dist_line_style = '--'

# data loading options
main_data_dir = '/home/trevor/data/paper-data/bc-view-agnostic/bc_results'
angle_dir = main_data_dir + '/Angle_50_rollouts_per_seed'

main_exp_dir = '/media/trevor/Data/paper-data/bc-viewag'
exp_dir = main_exp_dir + '/figures/ood_comparisons'
#--------------------------------------------------------------------------------

def get_angle_acc(exp_name, percentile):
    all_res_dir = angle_dir + '/' + exp_name + '/all_results.pkl'
    recorder = pickle.load(open(all_res_dir, 'rb'))
    angle_acc = np.array(recorder.per_ep_group_data['avg_suc'])
    angle_acc = angle_acc.reshape(12, -1).T
    mean, lower, upper = plot_utils.get_mean_std(angle_acc, percentile)
    return mean, lower, upper

def plot_angles(fix_name, multi_name, title, percentile, col, row=None):
    exp_name = {0:'Lift', 1:'Stack', 2:'PickAndInsert', 3:'Door'}
    f_mean, f_lower, f_upper = get_angle_acc(fix_name, percentile)
    m_mean, m_lower, m_upper = get_angle_acc(multi_name, percentile)

    f_error = np.abs(f_mean - f_lower)
    m_error = np.abs(m_mean - m_lower)

    cmap = plt.get_cmap("tab10")
    x = np.arange(f_mean.shape[0])
    width = 0.5
    if row is None:
        axis = axes[col]
    else:
        axis = axes[row, col]
    axis.grid(alpha=0.5)
    axis.bar(x+width/2, f_mean, width, color = cmap(0), alpha=0.5, yerr = f_error, ecolor=cmap(0), capsize=1.5)
    axis.bar(x+3*width/2, m_mean, width, color = cmap(1), alpha=0.5, yerr = m_error, ecolor=cmap(1), capsize=1.5)
    axis.set_title(title, fontsize=font_size - 6)
    axis.set_xticks(np.arange(13))
    axis.set_xticklabels(['-.8', '', '-.6', '', '-.4', '', '-.2', '', '0', '', '.2', '', '.4'])
    # axis.set_xticklabels((r"$-\frac{4\pi}{16}$", '', r"$-\frac{3\pi}{16}$", '', r"$-\frac{2\pi}{16}$", '',
    #                            r"$-\frac{1\pi}{16}$", '', r"$0$", '', r"$\frac{1\pi}{16}$", '',
    #                            r"$\frac{2\pi}{16}$"))
    axis.xaxis.set_tick_params(labelsize=font_size - 10)
    axis.yaxis.set_tick_params(labelsize=font_size - 10)
    axis.set_yticks(np.arange(0, 1.1, 0.2))
    axis.set_ylim([0.0, 1.0])

    # add vertical lines for showing ranges used experimentally
    axis.axvline(2, alpha=1.0, ls=dist_line_style, lw=dist_line_width, c=dist_line_color)
    axis.axvline(10, alpha=1.0, ls=dist_line_style, lw=dist_line_width, c=dist_line_color)


plot_utils.setup_pretty_plotting()

# Plotting the performance by angle graph
fig, axes = plt.subplots(nrows=plt_shape[0], ncols=plt_shape[1],
                         figsize=[plot_width * plt_shape[0], plot_height * plt_shape[1]])

if plt_shape[0] == 1:
    row_col = list(zip([None * 4], range(4)))
else:
    row_col = [[0, 0], [0, 1], [1, 0], [1, 1]]

env_names = [["ThingLiftXYZStateMultiview_100_fix", "ThingLiftXYZStateMultiview_100_multi"],
             ["ThingStackSameMultiviewV2_100_fix", "ThingStackSameMultiviewV2_100_multi"],
             ["ThingPickAndInsertSucDoneMultiview_100_fix", "ThingPickAndInsertSucDoneMultiview_100_multi"],
             ["ThingDoorMultiview_100_fix", "ThingDoorMultiview_100_multi"]]
env_titles = ['LiftSim', 'StackSim', 'PickAndInsertSim', 'DoorSim']
for envs, env_title, row_col_env in zip(env_names, env_titles, row_col):
    plot_angles(envs[0], envs[1], env_title, interval_percentile, row=row_col_env[0], col=row_col_env[1])

# x and y axis labels
# plot_utils.add_x_y_labels_multiplot(fig, r"Base $b_\phi$ Range (rad)", "Success Rate", font_size - 6, x_pad=10, y_pad=6)
plot_utils.add_x_y_labels_multiplot(fig, r"Base $b_\phi$ Range (rad)", "Success Rate", font_size - 6, x_pad=5, y_pad=6)

cmap = plt.get_cmap("tab10")  # Color scheme

# Legend
handles = [plt.Rectangle((0,0),1,1, color=cmap(0)), plt.Rectangle((0, 0),1,1,color=cmap(1)),
           plt.Line2D([0], [0], linewidth=dist_line_width, linestyle=dist_line_style, color=dist_line_color)]
# fig.legend(handles, ['Fixed-base Policy', 'Multiview Policy', 'Training Distribution Bounds'],
fig.legend(handles, [r'$\pi_f$', r'$\pi_m$', 'Training Distribution Bounds'],
           # loc="lower center", ncol=2, fontsize=font_size-7,
           loc="lower center", ncol=3, fontsize=font_size-7,
           # fancybox=True, shadow=True, bbox_to_anchor=(0.55, -0.015))
           # fancybox=True, shadow=True, bbox_to_anchor=(0.53, -0.09))  # old
           fancybox=True, shadow=True, bbox_to_anchor=(0.53, -0.03))

plt.tight_layout()

os.makedirs(exp_dir, exist_ok=True)
fig.savefig(exp_dir + '/ood.pdf', bbox_inches='tight')
