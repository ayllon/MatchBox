#!/usr/bin/env python3
# coding: utf-8
import logging
import sys
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas

from matchbox.util.loaders import load_datasets

logger = logging.getLogger('ds-summary')


def max_ind_by_name(d1, d2):
    return np.in1d(d1.columns, d2.columns).sum()


def main():
    """
    Entry point
    """
    # Basic setup
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.basicConfig(format='%(asctime)s %(name)15.15s %(levelname)s\t%(message)s',
                        level=logging.INFO, stream=sys.stderr)

    # Arguments
    parser = ArgumentParser(description='Display some summary statistics about a dataset')
    parser.add_argument('--seed', type=int, default=0, help='Initial random seed')
    parser.add_argument('--sample-size', type=int, default=200, help='Sample size')
    parser.add_argument('data1', metavar='DATA1', help='Dataset 1')
    parser.add_argument('data2', metavar='DATA2', help='Dataset 2')
    args = parser.parse_args()

    # Initialize the random state
    random_generator = np.random.MT19937(args.seed)

    # Load dataset
    logging.info('Loading datasets')
    datasets = load_datasets([args.data1, args.data2], filter_nan='column')

    # Display general summary
    print('Rows:', ' + '.join([str(len(df)) for _, df in datasets]))
    print('Columns:', ' + '.join([str(len(df.columns)) for _, df in datasets]))
    print('Max. IND:', max_ind_by_name(datasets[0][1], datasets[1][1]))


if __name__ == '__main__':
    main()
