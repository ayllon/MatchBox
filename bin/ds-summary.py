#!/usr/bin/env python3
# coding: utf-8
import logging
import os
import sys
import warnings
from argparse import ArgumentParser
from typing import List, Tuple, Dict

import numpy as np
import pandas
from pandas import DataFrame

from matchbox.uintersect import UIntersectFinder

try:
    import matchbox
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '..'))

from matchbox.util.loaders import load_datasets

logger = logging.getLogger('ds-summary')


def max_ind_by_name(d1: DataFrame, d2: DataFrame) -> int:
    """
    Return how many matching names there are between two datasets
    """
    return np.in1d(d1.columns, d2.columns).sum()


def uind_histogram(dataframes: List[Tuple[str, DataFrame]], alphas: List[float]) -> Dict[
    float, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute an histogram of how many matching columns each of the columns of the first dataframe
    has on the second one.

    Returns
    -------
    A dictionary where the key matches the significance levels passed as parameter,
    and the value is a pair with the binning and the counts
    """
    # UINDs have as an attribute the significance level, so
    # just compute the minimum, and then filter for the rest
    logger.info('Computing UIND')
    uind_finder = UIntersectFinder(method='ks')
    for df_name, df in dataframes:
        uind_finder.add(df_name, df)
    uinds = uind_finder(alpha=min(alphas), no_symmetric=True)
    logger.info('Computing histograms')
    histograms = {}
    for a in alphas:
        uind_subset = [u for u in uinds if u.confidence >= a]
        matches = {c: 0 for c in dataframes[0][1].columns}
        for u in uind_subset:
            c = u.lhs.attr_names[0]
            matches[c] += 1
        match_count = list(matches.values())
        bins = np.arange(0, max(match_count) + 1)
        edges = np.concatenate([[-0.5], bins + 0.5])
        hist, edges = np.histogram(match_count, bins=edges)
        histograms[a] = DataFrame(data=hist.reshape(1, -1), columns=bins)
    return histograms


def main():
    """
    Entry point
    """
    # Basic setup
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Arguments
    parser = ArgumentParser(description='Display some summary statistics about a dataset')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--seed', type=int, default=0, help='Initial random seed')
    parser.add_argument('--sample-size', type=int, default=200, help='Sample size')
    parser.add_argument('--uind-alphas', type=float, nargs='+', default=[0.05, 0.1],
                        help='Significance level for the unary IND tests')
    parser.add_argument('data1', metavar='DATA1', help='Dataset 1')
    parser.add_argument('data2', metavar='DATA2', help='Dataset 2')
    args = parser.parse_args()

    # Logging
    logging.basicConfig(format='%(asctime)s %(name)15.15s %(levelname)s\t%(message)s',
                        level=logging.DEBUG if args.debug else logging.INFO, stream=sys.stderr)

    # Initialize the random state
    random_generator = np.random.MT19937(args.seed)

    # Load dataset
    logging.info('Loading datasets')
    datasets = load_datasets([args.data1, args.data2])

    # Display general summary
    print('Rows:', ' + '.join([str(len(df)) for _, df in datasets]))
    print('Columns:', ' + '.join([str(len(df.columns)) for _, df in datasets]))
    print('Max. IND:', max_ind_by_name(datasets[0][1], datasets[1][1]))

    # Select samples
    logger.info('Using %d samples', args.sample_size)
    with pandas.option_context('mode.use_inf_as_na', True):
        samples = [
            (name, df.sample(args.sample_size, replace=True, random_state=random_generator))
            for name, df in datasets
        ]
    # Get the histogram for matching UINDs
    uind_hist = uind_histogram(samples, alphas=args.uind_alphas)
    for a in args.uind_alphas:
        print('Matching counts for', a)
        print('\t' + uind_hist[a].to_string().replace('\n', '\n\t'))


if __name__ == '__main__':
    main()
