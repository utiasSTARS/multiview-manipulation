import numpy as np
import scipy.stats
from matplotlib import pyplot as plt


def get_means_lowers_uppers(data, num_dem, seeds, interval_percentile):
    # some data files go 200 to 25/50, others go 25/50 to 200, so check for that
    first_num_dem = data[0, 1]
    if first_num_dem == 200:
        num_dem = num_dem[::-1]

    means = []; lowers = []; uppers = []
    for n_i, n in enumerate(num_dem):
        suc_rates = []
        for s_i, s in enumerate(seeds):
            data_row = s_i + n_i * len(seeds)
            if data[data_row][0] != s:
                print("Seed was wrong, script expected %d, file had %d" % (s, data[data_row][0]))
                import ipdb; ipdb.set_trace()
            if data[data_row][1] != n:
                print("Num dem was wrong, script expected %d, file had %d" % (n, data[data_row][1]))
                import ipdb; ipdb.set_trace()
            suc_rates.append(data[data_row][2])
        mean, lower, upper = plot_utils.get_mean_std(suc_rates, interval_percentile)
        means.append(mean); lowers.append(lower); uppers.append(upper)
    return num_dem, means, lowers, uppers


def plot_mean_and_std(axis, x_vals, means, lowers, uppers, color, yticks, xticks, ylim,
                      marker='.', extras=True, labelsize=10, title=None):
    if extras:
        axis.grid(alpha=0.5)
        axis.xaxis.set_tick_params(labelsize=labelsize, bottom=True, labelbottom=True)
        axis.yaxis.set_tick_params(labelsize=labelsize, left=True, labelleft=True)
    if title is not None:
        axis.set_title(title, fontsize=labelsize + 4)
    line = axis.plot(x_vals, means, color=color, marker=marker)
    axis.fill_between(x_vals, lowers, uppers, facecolor=color, alpha=0.5)
    axis.set_yticks(yticks)
    axis.set_xticks(xticks)
    axis.set_ylim(ylim)
    # axis.set_xlim(xlim)
    return line


def plot_four_conds(axes, env_str, env_i, wide_fig, label_size, num_dems_x,
                    mm_mean, mm_lower, mm_upper,
                    mf_mean, mf_lower, mf_upper,
                    fm_mean, fm_lower, fm_upper,
                    ff_mean, ff_lower, ff_upper
                    ):
    if wide_fig:
        col = env_i
        row = 0
    else:
        col = 0
        row = env_i

    cmap = plt.get_cmap("tab10")
    axes[row, col].grid(alpha=0.5)
    axes[row, col].plot(num_dems_x, ff_mean, color=cmap(0), marker='.')
    axes[row, col].fill_between(num_dems_x, ff_lower, ff_upper, facecolor=cmap(0), alpha=0.5)
    axes[row, col].plot(num_dems_x, mf_mean, color=cmap(1), marker='.')
    axes[row, col].fill_between(num_dems_x, mf_lower, mf_upper, facecolor=cmap(1), alpha=0.5)
    axes[row, col].xaxis.set_tick_params(labelsize=label_size, length=2, bottom=True, labelbottom=True)
    axes[row, col].yaxis.set_tick_params(labelsize=label_size, length=2, left=True, labelleft=True)
    if 'Sim' in env_str:
        axes[row, col].set_xticks(np.arange(25, 210, 25))
    else:
        axes[row, col].set_xticks(np.arange(50, 210, 50))
    if wide_fig:
        axes[row, col].set_title(env_str, fontsize=label_size + 4)
    else:
        axes[row, col].set_ylabel(env_str, fontsize=label_size + 4)

    axes[row, col].set_ylim([-.05, 1.05])
    axes[row, col].set_yticks(np.arange(0, 1.1, .25))

    if wide_fig:
        col = env_i
        row = 1
    else:
        col = 1
        row = env_i

    axes[row, col].grid(alpha=0.5)
    f_line = axes[row, col].plot(num_dems_x, fm_mean, color=cmap(0), marker='.')
    axes[row, col].fill_between(num_dems_x, fm_lower, fm_upper, facecolor=cmap(0), alpha=0.5)
    m_line = axes[row, col].plot(num_dems_x, mm_mean, color=cmap(1), marker='.')
    axes[row, col].fill_between(num_dems_x, mm_lower, mm_upper, facecolor=cmap(1), alpha=0.5)
    axes[row, col].xaxis.set_tick_params(labelsize=label_size, length=2, bottom=True, labelbottom=True)
    axes[row, col].yaxis.set_tick_params(labelsize=label_size, length=2, left=True, labelleft=True)
    if 'Sim' in env_str:
        axes[row, col].set_xticks(np.arange(25, 210, 25))
    else:
        axes[row, col].set_xticks(np.arange(50, 210, 50))

    axes[row, col].set_ylim([-.05, 1.05])
    axes[row, col].set_yticks(np.arange(0, 1.1, .25))

    return f_line, m_line


def add_x_y_labels_multiplot(fig_handle, x_label, y_label, fontsize, x_pad=0, y_pad=0):
    ax = fig_handle.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(x_label, labelpad=x_pad, fontsize=fontsize)
    # ax.xaxis.set_label_coords(0.6, -0.1)
    plt.ylabel(y_label, labelpad=y_pad, fontsize=fontsize)


def get_mean_std(acc, percentile):
    mean = np.array(acc).mean(axis=0)
    std = np.std(acc, axis=0)

    # this doesn't actually give what we want -- this gives the sigma for Pr(X < sigma) < percentile / 100
    # norm_percentile = scipy.stats.norm.ppf(percentile / 100)

    # instead, we want sigma for Pr(-sigma < X < sigma) < percentile / 100, which is
    # (1 - 2 * (1 - cdf(sigma))) = percentile / 100 -- aka the 68-95-99.7 empirical rule
    norm_percentile = scipy.stats.norm.ppf(1 - (1 - percentile / 100) / 2)

    return mean, mean - std * norm_percentile, mean + std * norm_percentile


def setup_pretty_plotting():
    # pretty plotting stuff
    font_params = {
    "font.family": "serif",
    "font.serif": "Times",
    "text.usetex": True,
    "pgf.rcfonts": False
    }
    plt.rcParams.update(font_params)