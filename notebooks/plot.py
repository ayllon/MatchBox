from typing import Union, Tuple, Any, Mapping, Iterable

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib import cm, gridspec
from matplotlib.colors import Colormap, LinearSegmentedColormap

discrete_grays = LinearSegmentedColormap.from_list('DiscreteGrays', [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0)], N=5)


class StyleCycler:
    def __init__(self, markers, lines, colors):
        self.markers = markers
        self.lines = lines
        self.colors = colors

    def __iter__(self):
        m = itertools.cycle(self.markers)
        l = itertools.cycle(self.lines)
        c = self.colors()
        while True:
            yield next(m), next(l), next(c)['color']


def readable_key(Lambda, gamma, grow):
    if gamma > 1:
        return f'$\\Lambda={Lambda}$ $\\gamma=0$ {"grow" if grow else ""}'
    elif gamma == 1:
        return f'$\\Lambda={Lambda}$ $\\gamma=1-\\alpha$ {"grow" if grow else ""}'
    return f'$\\Lambda={Lambda}$ $\\gamma=1-{gamma}\\alpha$ {"grow" if grow else ""}'


def plot_violin(df: pandas.DataFrame, key: str, color: Union[str, Tuple], group_by: str, ax: plt.Axes = None):
    """
    Draw a violin plot, showing the distribution of the field 'key', grouped by 'group_by'

    Parameters
    ----------
    df : DataFrame
        Output from benchmark.py
    key : str
        Show the distribution for this value
    color :
        Using this color
    group_by :
        Grouped by this field (one violin per unique value)
    ax :
        Axes where to draw the violin plot
    """
    if ax is None:
        ax = plt.gca()

    groups = df[group_by].unique()
    y = [df[df[group_by] == g][key] for g in groups]
    x = np.arange(len(groups))

    draws = ax.violinplot(y, x, widths=0.5, showmedians=False, showextrema=False)
    q = np.array([np.quantile(v, [0.25, 0.5, 0.75]) for v in y])
    ax.vlines(x, q[:, 0], q[:, 2], color=color, lw=4, alpha=0.5)
    ax.scatter(x, q[:, 1], marker='.', color='white', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)

    for d in draws['bodies']:
        d.set_facecolor(color)


def plot_comparison(find2: pandas.DataFrame, findq: Mapping[Any, pandas.DataFrame],
                    findq_subset: Iterable = None, max_ind_column: str = None,
                    fig: plt.Figure = None, cmap: Union[Colormap, str] = discrete_grays):
    """
    Generate a plot comparing the results of find2 and findq

    Parameters
    ----------
    find2 : DataFrame
        Results of Find2
    findq : Dictionary of DataFrame
        Results of FindQ
    findq_subset :
        Select a subset of the parametrization of FindQ
    max_ind_column :
        Name of the column with the maximum number of IND. If None, it will use the first
        that starts with `max_`
    fig :
        Figure where to plot. If None, one will be created.
    cmap :
        Colormap to use for each column. Defaults to a gray scale.
    """
    if not max_ind_column:
        for c in find2.columns:
            if c.startswith('max_'):
                max_ind_column = c
                break
        if not max_ind_column:
            raise KeyError('Could not figure out the column with the maximum arity found')

    if findq_subset is not None:
        findq = dict([(key, findq[key]) for key in findq_subset])

    if fig is None:
        fig = plt.figure(figsize=(9, 5))

    ncolumns = 1 + len(findq)

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=ncolumns))

    grid = gridspec.GridSpec(4, ncolumns)
    grid.update(wspace=0.0, hspace=0.0)

    # Plot find2 as reference
    ax_time = fig.add_subplot(grid[0])
    plot_violin(find2, 'time', color=sm.to_rgba(0), group_by='bootstrap_alpha', ax=ax_time)
    ax_time.set_title('Find2')
    ax_time.set_ylabel('Time (s)')
    ax_time.yaxis.grid(True)
    ax_time.set_yscale('log')

    ax_nind = fig.add_subplot(grid[1 * ncolumns])
    plot_violin(find2, f'{max_ind_column}', color=sm.to_rgba(0), group_by='bootstrap_alpha', ax=ax_nind)
    ax_nind.set_ylabel('Maximal nIND')
    ax_nind.yaxis.grid(True)

    ax_tests = fig.add_subplot(grid[2 * ncolumns])
    plot_violin(find2, 'tests', color=sm.to_rgba(0), group_by='bootstrap_alpha', ax=ax_tests)
    ax_tests.set_ylabel('Tests')
    ax_tests.set_yscale('log')
    ax_tests.yaxis.grid(True)

    ax_uind = fig.add_subplot(grid[3 * ncolumns])
    plot_violin(find2, 'unique_ind', color=sm.to_rgba(0), group_by='bootstrap_alpha', ax=ax_uind)
    ax_uind.set_ylabel('Unique EDD')
    ax_uind.set_xlabel('Initial $\\alpha$')
    ax_uind.yaxis.grid(True)

    # Plot findq
    for i, (p, data) in enumerate(findq.items(), 1):
        lambd, gamma, grow = p
        color = sm.to_rgba(i)

        axt = fig.add_subplot(grid[i], sharex=ax_time, sharey=ax_time)
        plot_violin(data, 'time', color=color, group_by='bootstrap_alpha', ax=axt)
        axt.set_title(readable_key(lambd, gamma, grow))
        plt.setp(axt.get_yticklabels(), visible=False)
        axt.yaxis.grid(True)

        axn2 = fig.add_subplot(grid[1 * ncolumns + i], sharey=ax_nind, sharex=ax_nind)
        plot_violin(data, f'{max_ind_column}', color=color, group_by='bootstrap_alpha', ax=axn2)
        plt.setp(axn2.get_yticklabels(), visible=False)
        axn2.yaxis.grid(True)

        axtt = fig.add_subplot(grid[2 * ncolumns + i], sharey=ax_tests, sharex=ax_tests)
        plot_violin(data, 'tests', color=color, group_by='bootstrap_alpha', ax=axtt)
        axtt.yaxis.grid(True)
        plt.setp(axtt.get_yticklabels(), visible=False)
        axtt.set_xlabel('Initial $\\alpha$')

        axni = fig.add_subplot(grid[3 * ncolumns + i], sharey=ax_uind, sharex=ax_uind)
        plot_violin(data, 'unique_ind', color=color, group_by='bootstrap_alpha', ax=axni)
        axni.yaxis.grid(True)
        plt.setp(axni.get_yticklabels(), visible=False)
        axni.set_xlabel('Initial $\\alpha$')

        # fig.suptitle(f'{max_ind_column}')
        fig.tight_layout()
        fig.align_ylabels([ax_time, ax_nind, ax_tests, ax_uind])
    return fig


def bootstrap_samples_mean(data: pandas.DataFrame, size: int = None, samples: int = 1000):
    """
    Compute the samples mean via bootstrapping
    """
    if size is None:
        size = len(data)
    samples = [np.random.choice(data, size=size, replace=True) for _ in range(samples)]
    return np.mean(samples, axis=1)


def plot_confidence(ax: plt.Axes, i: int, data: pandas.DataFrame, ref, **kwargs):
    ref_sample = bootstrap_samples_mean(ref)
    sample = 100 * ((bootstrap_samples_mean(data) - ref_sample) / ref_sample)
    avg, std = np.average(sample), 1.96 * np.std(sample)
    ax.errorbar([i], [avg], yerr=std, capsize=10, **kwargs)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter('{x:.0f} \\%')
    for l in ax.get_yticklabels():
        l.set_fontsize(12)


def bootstrap_plot(find2: pandas.DataFrame, findq: Mapping[Any, pandas.DataFrame],
                   findq_subset: Iterable = None, max_ind_column: str = None, fig: plt.Figure = None,
                   alphas=[0.05, 0.10, 0.15], title=None):
    if title is None:
        title = 'Distribution of the samples mean'
    if not max_ind_column:
        for c in find2.columns:
            if c.startswith('max_'):
                max_ind_column = c
                break
    if not max_ind_column:
        raise KeyError('Could not figure out the column with the maximum arity found')
    if fig is None:
        fig = plt.figure(figsize=(18, 2.5 * len(alphas)))

    cycler = StyleCycler(['o', 's', 'D', '*'], ['--'],  plt.rcParams['axes.prop_cycle'])

    # Mask out exact == 0 and timeouts
    e0mask = find2['exact'] > 0 & ~find2['timeout']
    find2 = find2[e0mask]

    # Ratios
    ratio_f2 = find2[max_ind_column] / find2['exact']

    # Figure grid
    grid = gridspec.GridSpec(nrows=len(alphas), ncols=4)# hspace=0)

    ax_ratio = None

    # For each alpha
    for ia, alpha in enumerate(alphas):
        # Axes
        ax_ratio = fig.add_subplot(grid[ia, 0], sharex=ax_ratio)
        ax_nind = fig.add_subplot(grid[ia, 1], sharex=ax_ratio)
        ax_tests = fig.add_subplot(grid[ia, 2], sharex=ax_ratio)
        ax_time = fig.add_subplot(grid[ia, 3], sharex=ax_ratio)

        for ax in [ax_ratio, ax_nind, ax_tests, ax_time]:
            ax.axhline(0, linestyle='--', c='red')

        f2mask = find2['bootstrap_alpha'] == alpha
        fqmask = dict()
        for k in findq_subset:
            fqmask[k] = (findq[k]['bootstrap_alpha'] == alpha) & (findq[k]['exact'] > 0) & (~findq[k]['timeout'])

        ref_ratio = ratio_f2[f2mask]
        ref_time = find2['time'][f2mask]
        ref_tests = find2['tests'][f2mask]
        ref_unique = find2['unique_ind'][f2mask]

        # For each findq parametrization
        for (i, k), (marker, _, color) in zip(enumerate(findq_subset, start=1), cycler):
            v = findq[k][fqmask[k]]
            label = readable_key(*k)
            plot_confidence(ax_ratio, i, v[max_ind_column] / v['exact'], ref=ref_ratio, label=label, marker=marker, color=color)
            plot_confidence(ax_time, i, v['time'], ref=ref_time, label=label, marker=marker, color=color)
            plot_confidence(ax_tests, i, v['tests'], ref=ref_tests, label=label, marker=marker, color=color)
            plot_confidence(ax_nind, i, v['unique_ind'], ref=ref_unique, label=label, marker=marker, color=color)

        ax_ratio.set_ylabel(f'$\\alpha = {alpha}$')

        if ia == 0:
            ax_time.legend()
            ax_ratio.set_title('Ratio')
            ax_ratio.title.set_fontsize(18)
            ax_time.set_title('Time')
            ax_time.title.set_fontsize(18)
            ax_tests.set_title('Tests')
            ax_tests.title.set_fontsize(18)
            ax_nind.set_title('Unique EDD')
            ax_nind.title.set_fontsize(18)
            #ax_ratio.set_xlim(0.5, len(findq_subset) + 0.5)

    fig.tight_layout(w_pad=1, h_pad=0)
    return fig
