import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

from multiview_manipulation import plotting as plot_utils, utils as bc_viewag_plot_utils

# CONFIG
#-------------------------------------------------------------------------------
# plot options
wide_full_comp = True  # wide is 2 rows by 5 cols, otherwise 5 by 2
font_size = 22
interval_percentile = 95
plot_width = 3.2
plot_height = 2.4

# data loading options
main_data_dir = '/home/trevor/data/paper-data/bc-view-agnostic/bc_results'
bc_folders_file = 'bc_result_folder_names.yaml'
full_comp_envs = ['LiftSim', 'StackSim', 'PickAndInsertSim', 'DoorSim', 'DoorReal']
conditions = ['mm', 'mf', 'fm', 'ff']
mult_only_envs = ['PickAndInsertReal', 'DrawerReal']
mult_only_envs_vertical = True
sim_bc_seeds = [1, 2, 3, 4, 5]
sim_num_dem = range(25, 201, 25)
real_bc_seeds = [1, 2, 3]
real_num_dem = range(50, 201, 50)
results_filename = 'all_results.npz'

main_exp_dir = '/media/trevor/Data/paper-data/bc-viewag'
exp_dir = main_exp_dir + '/figures/bc_results' # + '/' + datetime.now().strftime("%y-%m-%d_%H-%M-%S")
#--------------------------------------------------------------------------------

plot_utils.setup_pretty_plotting()
bc_folders = yaml.load(open(bc_folders_file, 'r'), yaml.Loader)

# Full comparison fig (4 conditions) -----------------------------------------------------------------
if wide_full_comp:
    fig, axes = plt.subplots(nrows=2, ncols=len(full_comp_envs), # sharex=True, sharey=True,
                             figsize=[plot_width * len(full_comp_envs), plot_height * 2])
    axes[0, 0].set_ylabel("Fixed-base Env", labelpad=20, fontsize=font_size - 6)
    axes[1, 0].set_ylabel("Multiview Env", labelpad=20, fontsize=font_size - 6)
else:
    fig, axes = plt.subplots(nrows=len(full_comp_envs), ncols=2, # sharex=True, sharey=True,
                             figsize=[plot_width * 2, plot_height * len(full_comp_envs)])
    axes[0, 0].set_title("Fixed Env", labelpad=20, fontsize=font_size - 6)
    axes[0, 1].set_title("Multiview Env", labelpad=20, fontsize=font_size - 6)

full_comp_data = dict()
for env_i, env in enumerate(full_comp_envs):
    full_comp_data[env] = {c: 0 for c in conditions}

    for cond_i, cond in enumerate(conditions):
        data = np.load(main_data_dir + '/' + bc_folders[env][cond_i] + '/' + results_filename)['per_ep_group']

        if 'Sim' in env:
            seeds = sim_bc_seeds
            num_dem = sim_num_dem
        else:
            seeds = real_bc_seeds
            num_dem = real_num_dem

        num_dem, means, uppers, lowers = bc_viewag_plot_utils.get_means_lowers_uppers(data, num_dem, seeds, interval_percentile)
        full_comp_data[env][cond] = dict(means=means, lowers=lowers, uppers=uppers)

    # plot now that all data collected
    fcd = full_comp_data
    f_line, m_line = bc_viewag_plot_utils.plot_four_conds(axes, env, env_i, wide_full_comp, font_size - 10, num_dem,
                                         fcd[env]['mm']['means'], fcd[env]['mm']['lowers'], fcd[env]['mm']['uppers'],
                                         fcd[env]['mf']['means'], fcd[env]['mf']['lowers'], fcd[env]['mf']['uppers'],
                                         fcd[env]['fm']['means'], fcd[env]['fm']['lowers'], fcd[env]['fm']['uppers'],
                                         fcd[env]['ff']['means'], fcd[env]['ff']['lowers'], fcd[env]['ff']['uppers'])

fig.legend([m_line, f_line],
           labels=["Fixed-base Policy", "Multiview Policy"],
           # labels=[r"$\pi_f$", r"$\pi_m$"],

           ncol=2,
           fancybox=True,
           shadow=True,
           fontsize=font_size - 6,
           # loc="lower left",              # on figure
           # bbox_to_anchor=(0.1, 0.175),   # on figure
           loc="lower right",              # bottom right  -- this is the original one
           # bbox_to_anchor=(0.05, 0.015),   # bottom right
           # loc="lower left",              # bottom left
           # bbox_to_anchor=(0.05, 0.015),  # bottom left
           # loc="lower center",            # center under
           # bbox_to_anchor=(0.535, -0.05)  # center under
           )

ax = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.xlabel("Number of Training Demonstrations", fontsize=font_size-6)
plt.xlabel("Number of Training Demonstrations", fontsize=font_size-6)
# ax.xaxis.set_label_coords(0.6, -0.1)
plt.ylabel("Success Rate", labelpad=10, fontsize=font_size-6)

plt.tight_layout()

os.makedirs(exp_dir, exist_ok=True)
fig.savefig(exp_dir + '/full_comp_success.pdf', bbox_inches='tight')


# Multiview suc only fig -----------------------------------------------------------------
if mult_only_envs_vertical:
    fig, axes = plt.subplots(nrows=len(mult_only_envs), ncols=1,  # sharex=True, sharey=True,
                             figsize=[plot_width, (plot_height * len(mult_only_envs)) + .5])
else:
    fig, axes = plt.subplots(nrows=1, ncols=len(mult_only_envs), # sharex=True, sharey=True,
                             figsize=[plot_width * len(mult_only_envs), plot_height + .5])
ax = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.xlabel("Number of Training Demonstrations", fontsize=font_size-6)
plt.xlabel("Number of Training Demonstrations", fontsize=font_size-6)
# ax.xaxis.set_label_coords(0.6, -0.1)
plt.ylabel("Success Rate", labelpad=10, fontsize=font_size-6)

mult_only_data = dict()
cmap = plt.get_cmap("tab10")
for env_i, env in enumerate(mult_only_envs):
    mult_only_data[env] = 0
    data = np.load(main_data_dir + '/' + bc_folders[env][0] + '/' + results_filename)['per_ep_group']
    if 'Sim' in env:
        seeds = sim_bc_seeds
        num_dem = sim_num_dem
    else:
        seeds = real_bc_seeds
        num_dem = real_num_dem

    num_dem, means, uppers, lowers = bc_viewag_plot_utils.get_means_lowers_uppers(data, num_dem, seeds, interval_percentile)
    line = bc_viewag_plot_utils.plot_mean_and_std(axes[env_i], num_dem, means, lowers, uppers, cmap(1),
                                                  yticks=np.arange(0, 1.1, .25), xticks=np.arange(50, 210, 50),
                                                  ylim=[-.05, 1.05], labelsize=font_size-10, title=env)

# fig.legend([m_line, f_line],
#            labels=["Multiview Policy"],
#            ncol=1,
#            fancybox=True,
#            shadow=True,
#            fontsize=font_size - 6,
#            loc="right",              # bottom right
#            bbox_to_anchor=(0.96, 0.4),   # bottom right
#            # loc="lower left",              # bottom left
#            # bbox_to_anchor=(0.05, 0.015),  # bottom left
#            # loc="lower center",            # center under
#            # bbox_to_anchor=(0.535, -0.05)  # center under
#            )

plt.tight_layout()

fig.savefig(exp_dir + '/mult_only_envs.pdf', bbox_inches='tight')
# plt.show()
