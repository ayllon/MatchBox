from typing import Union, Tuple, Any, Mapping, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib import cm, gridspec
from matplotlib.colors import Colormap, LinearSegmentedColormap

discrete_grays = LinearSegmentedColormap.from_list('DiscreteGrays', [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0)], N=5)


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
        fig = plt.figure()

    ncolumns = 1 + len(findq)

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=ncolumns))

    grid = gridspec.GridSpec(3, ncolumns)
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
    ax_tests.set_ylabel('Number of tests')
    ax_tests.set_xlabel('Initial $\\alpha$')
    ax_tests.set_yscale('log')
    ax_tests.yaxis.grid(True)

    # Plot findq
    for i, (p, data) in enumerate(findq.items(), 1):
        lambd, gamma = p
        color = sm.to_rgba(i)

        axt = fig.add_subplot(grid[i], sharex=ax_time, sharey=ax_time)
        plot_violin(data, 'time', color=color, group_by='bootstrap_alpha', ax=axt)
        if gamma == 1:
            axt.set_title(f'$\\Lambda={lambd}$ $\\gamma=1-\\alpha$')
        else:
            axt.set_title(f'$\\Lambda={lambd}$ $\\gamma=1-{gamma}\\alpha$')
        plt.setp(axt.get_yticklabels(), visible=False)
        axt.yaxis.grid(True)

        axn2 = fig.add_subplot(grid[1 * ncolumns + i], sharey=ax_nind, sharex=ax_nind)
        plot_violin(data, f'{max_ind_column}', color=color, group_by='bootstrap_alpha', ax=axn2)
        plt.setp(axn2.get_yticklabels(), visible=False)
        axn2.yaxis.grid(True)

        axtt = fig.add_subplot(grid[2 * ncolumns + i], sharey=ax_tests, sharex=ax_tests)
        plot_violin(data, 'tests', color=color, group_by='bootstrap_alpha', ax=axtt)
        axtt.set_yscale('log')
        axtt.yaxis.grid(True)
        plt.setp(axtt.get_yticklabels(), visible=False)
        axtt.set_xlabel('Initial $\\alpha$')

    fig.suptitle(f'{max_ind_column}')
    fig.tight_layout()
