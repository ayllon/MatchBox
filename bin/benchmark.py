#!/usr/bin/env python3
# coding: utf-8
import itertools
import logging
import os
import sys
import time
import uuid
import warnings
from argparse import ArgumentParser
from functools import reduce
from typing import List, Tuple, Set, Callable, Type, Iterable

import numpy as np
import pandas
from filelock import FileLock
from numpy.random import BitGenerator
from pandas import DataFrame

from matchbox.find2 import Find2
from matchbox.find_gamma import FindGamma
from matchbox.hypergraph import generate_graph
from matchbox.ind import Ind, unique_inds, find_max_arity_per_pair
from matchbox.tests import knn_test
from matchbox.util.callcounter import CallCounter
from matchbox.util.keel import parse_keel_file
from matchbox.util.plot import to_dot_file
from matchbox.util.timing import Timing

logger = logging.getLogger('benchmark')


def load_fits(path: str, ncols: int) -> DataFrame:
    """
    Load a FITS catalog. Only the first table HDU is supported.
    """
    from astropy.table import Table
    df = Table.read(path).to_pandas()
    if ncols:
        df = df[df.columns[:ncols]]
    # Need to mask "special" -99 values
    df.replace(-99, np.nan)
    return df


def load_csv(path: str, ncols: int) -> DataFrame:
    """
    Load a CSV file using pandas
    """
    return pandas.read_csv(path, usecols=range(ncols), skipinitialspace=True)


# List of supported formats and their loaders
_loaders = {
    '.fits': load_fits,
    '.dat': parse_keel_file,
    '.csv': load_csv
}


def load_datasets(paths: List[str], sample_size: int, random_state: BitGenerator,
                  ncols: int = None) -> List[Tuple[str, DataFrame]]:
    """
    Load a list of datasets from files

    Parameters
    ----------
    paths : list of strings
        Paths to the datasets
    sample_size : int
        Sample size to read from the file
    random_state : BitGenerator
        Used to set the initial random state of the sampler
    ncols : int
        Read only this number of columns (all if 0 or None)

    Returns
    -------
    out : List of tuples (name, dataframe)
    """
    dataframes = []
    for path in paths:
        logger.info('Loading %s', path)
        ext = os.path.splitext(path)[1]
        df = _loaders[ext](path, ncols)
        with pandas.option_context('mode.use_inf_as_na', True):
            # Drop columns that are all NaN or Inf
            df.dropna(axis=1, how='all', inplace=True)
            # Drop rows with at least one NaN or Inf
            df.dropna(axis=0, how='any', inplace=True)
        name = os.path.basename(path)
        logger.info('\t%d columns loaded', len(df.columns))
        dataframes.append((name, df.sample(sample_size, replace=True, random_state=random_state)))
    return dataframes


def generate_uind(dataframes: List[Tuple[str, DataFrame]], alpha: float) -> Set[Ind]:
    """
    Run the unary finder n the given dataframes

    Parameters
    ----------
    dataframes : list of dataframes
    alpha : float
        Significance level for the KS statistic

    Returns
    -------
    out : set of Ind
        Set of unary IND
    """
    from matchbox.uintersect import UIntersectFinder

    timing = Timing()
    uind_finder = UIntersectFinder(method='ks')
    for df_name, df in dataframes:
        uind_finder.add(df_name, df)
    with timing:
        uinds = uind_finder(alpha=alpha, no_symmetric=True)
    logger.info('Number of UIND: %d', len(uinds))
    logger.info('Took %.2f seconds', timing.elapsed)
    return uinds


def bootstrap_ind(uinds: Set[Ind], stop: int, alpha: float, test_method: Callable, test_args: dict) -> Set[Ind]:
    """
    Run MIND on the initial set of unary IND up to the level `stop`

    Parameters
    ----------
    uinds : Set of ind
        Initial unary IND
    stop : Run MIND up to this arity
    alpha : alpha
        Significance level for the statistical test used to validate the n-IND candidates
    test_method : Callable
        Statistical test for validating the n-IND candidates
    test_args : Dictionary
        Additional set of keyword parameters to pass to the test method

    Returns
    -------
    out : Set of Ind
        Validated set of n-IND
    """
    from matchbox.mind import Mind

    timing = Timing()
    mind_bootstrap = Mind(alpha=alpha, test=test_method, test_args=test_args)
    with timing:
        inds = mind_bootstrap(uinds, stop=stop)
    logger.info('Number of %d-IND: %d', stop, len(inds))
    logger.info('Took %.2f seconds', timing.elapsed)
    return inds


def run_finder(Finder: Type, alpha: float,
               bootstrap_arity: int, bootstrap_ind: Set[Ind], bootstrap_alphas: Iterable[float],
               exact: int, test_method: Callable, test_args: dict,
               output_dir: str, csv_name: str,
               kwargs_callback: Callable = lambda alpha: {}):
    """
    Wrap the execution of a n-IND finder, timing it and counting the number of tests checks

    Parameters
    ----------
    Finder : Type
        Type of the finder (i.e. Find2)
    alpha : float
        Significance level for the statistical test used to validate the n-IND candidates
    bootstrap_arity :
        Arity of the initial set of IND
    bootstrap_ind :
        Initial set of IND
    bootstrap_alphas : Iterable of floats
        Tests can be executed with different initial significance levels.
    exact : int
        Number of exact unary IND known (by name)
    test_method : Callable
        Statistical test for validating the n-IND candidates
    test_args : Dictionary
        Additional set of keyword parameters to pass to the test method
    output_dir : str
        Where to write the output of the tests
    csv_name : str
        Name of the output CSV file (relative to output_dir)
    kwargs_callback : Callable
        It can be used by the caller to adapt the parameters passed to the Finder
    """
    results = {
        'id': [], 'exact': [], 'alpha': [], 'time': [], 'tests': [], 'ind': [], 'unique_ind': []
    }

    finder_name = type(Finder).__name__

    for b_alpha in bootstrap_alphas:
        logger.info('%s starting with alpha=%.2f', finder_name, b_alpha)
        selected_ind = [ind for ind in bootstrap_ind if ind.confidence >= b_alpha]
        counted_test = CallCounter(test_method)

        finder = Finder(
            n=bootstrap_arity, alpha=alpha, test=counted_test, test_args=test_args,
            **kwargs_callback(b_alpha)
        )

        timing = Timing()
        with timing:
            nind = finder(selected_ind)

        unique_nind = unique_inds(nind)

        run_id = str(uuid.uuid1())
        results['id'].append(run_id)
        results['exact'].append(exact)
        results['alpha'].append(alpha)
        results['time'].append(timing.elapsed)
        results['tests'].append(counted_test.counter)
        results['ind'].append(len(nind))
        results['unique_ind'].append(len(unique_nind))

        max_inds = find_max_arity_per_pair(unique_nind)
        for pair, ind in max_inds.items():
            if f'max_{pair}' not in results:
                results[f'max_{pair}'] = list()
            results[f'max_{pair}'].append(ind.arity)

        nind_dir = os.path.join(output_dir, run_id[:2], run_id)
        os.makedirs(nind_dir)

        # Save a summary of the arities
        arities = [i.arity for i in unique_nind]
        bins, counts = np.unique(arities, return_counts=True)
        np.savetxt(os.path.join(nind_dir, 'histogram.txt'), np.stack([bins, counts]).astype(np.int32), fmt='%d')

        # Save all ind
        with open(os.path.join(nind_dir, 'nind.txt'), 'wt') as fd:
            for i in unique_nind:
                print(i.arity, str(i), file=fd)

    result = pandas.DataFrame(results)
    csv_path = os.path.join(output_dir, csv_name)
    with FileLock(csv_path + '.lock'):
        result.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))


def define_arguments() -> ArgumentParser:
    """
    Initialize an ArgumentParser
    """
    parser = ArgumentParser()
    parser.add_argument('--id', type=str, default=None,
                        help='Run identifier, defaults to a derived from the dataset file names')
    parser.add_argument('--output-dir', type=str, default='/tmp',
                        help='Write the generated output to this directory')
    parser.add_argument('--seed', type=int, default=time.time_ns(),
                        help='Initial random seed')
    parser.add_argument('--sample-size', type=int, default=200,
                        help='Sample size')
    parser.add_argument('-k', type=int, default=3,
                        help='Number of neighbors for the KNN test')
    parser.add_argument('-p', '--permutations', type=int, default=500,
                        help='Number of permutations for the KNN test')
    parser.add_argument('--uind-alpha', type=float, default=0.05,
                        help='Significance level for the unary IND tests (KS)')
    parser.add_argument('--nind-alpha', type=float, default=0.05,
                        help='Significance level for the n-IND tests (KNN)')
    parser.add_argument('--bootstrap-alpha', type=float, nargs='+', default=[0.05, 0.1, 0.15],
                        help='Significance levels for the bootstrapping tests (KNN)')
    parser.add_argument('--bootstrap-arity', type=int, default=2,
                        help='Run MIND up to this arity')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.01, 0.05, 0.1],
                        help='Significance level for the Hyper-geometric test on the degrees of the nodes')
    parser.add_argument('--gammas', type=float, nargs='+', default=[1., 1.1, 1.5],
                        help='Gamma factor for the number of missing edges on a quasi-clique')
    parser.add_argument('--columns', type=int, default=None,
                        help='Select a subset of the columns')
    parser.add_argument('--write-dot', action='store_true',
                        help='Write a dot file with the initial graph')
    parser.add_argument('data1', metavar='DATA1', help='Dataset 1')
    parser.add_argument('data2', metavar='DATA2', help='Dataset 2')
    return parser


def main():
    """
    Entry point
    """
    # Basic setup
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.basicConfig(format='%(asctime)s %(name)15.15s %(levelname)s\t%(message)s',
                        level=logging.INFO, stream=sys.stderr)
    logging.getLogger('filelock').setLevel(logging.WARN)

    # Parse arguments
    parser = define_arguments()
    args = parser.parse_args()

    # Initialize the random state
    random_generator = np.random.MT19937(args.seed)

    # Fill defaults
    if not args.id:
        args.id = os.path.basename(args.data1) + '_' + os.path.basename(args.data2)
    output_dir = os.path.join(args.output_dir, args.id)
    logger.info(f'Using output directory {output_dir}')

    # Create output directory if necessary
    os.makedirs(output_dir, exist_ok=True)

    # Load a sample from the input datasets
    samples = load_datasets([args.data1, args.data2], ncols=args.columns, sample_size=args.sample_size,
                            random_state=random_generator)

    # Initial set of unary IND
    logger.info('Looking for unary IND')
    uinds = generate_uind(samples, alpha=args.uind_alpha)
    uind_name_match = list(filter(lambda u: u.lhs.attr_names == u.rhs.attr_names, uinds))
    logger.info('%d unary IND found with matching names', len(uind_name_match))

    # Common statistical test setup
    test_args = dict(k=args.k, n_perm=args.permutations)
    test_method = knn_test

    # Bootstrap initial set of IND using mind
    # The individual methods can bootstrap themselves, but we want to have the initial conditions  as close as possible
    initial_ind = bootstrap_ind(
        uinds, stop=args.bootstrap_arity, alpha=min(args.bootstrap_alpha),
        test_method=test_method, test_args=test_args
    )
    induced_uind = reduce(frozenset.union, map(Ind.get_all_unary, initial_ind), frozenset())
    uind_name_match = list(filter(lambda u: u.lhs.attr_names == u.rhs.attr_names, induced_uind))
    logger.info('%d unary IND found with matching names after bootstrapping', len(uind_name_match))

    # Write dot file with the initial graph
    if args.write_dot:
        graph_dot_file = os.path.join(output_dir, f'{args.bootstrap_arity}-graph.dot')
        bootstrap_graph, _ = generate_graph(initial_ind)
        to_dot_file(bootstrap_graph.E, graph_dot_file)
        logger.info('Initial graph written to %s', graph_dot_file)

    # Benchmark find2
    run_finder(
        Find2, alpha=args.nind_alpha,
        bootstrap_arity=args.bootstrap_arity, bootstrap_ind=initial_ind, bootstrap_alphas=args.bootstrap_alpha,
        exact=len(uind_name_match), test_method=test_method, test_args=test_args,
        output_dir=output_dir, csv_name='find2.csv'
    )

    # Benchmark findg
    for lambd, gamma, in itertools.product(args.lambdas, args.gammas):
        if np.isinf(lambd) and np.isinf(gamma):
            continue
        logger.info('FindG lambda=%.2f, gamma=1 - %.2f * alpha', lambd, gamma)
        run_finder(
            FindGamma, alpha=args.nind_alpha,
            bootstrap_arity=args.bootstrap_arity, bootstrap_ind=initial_ind, bootstrap_alphas=args.bootstrap_alpha,
            exact=len(uind_name_match), test_method=test_method, test_args=test_args,
            output_dir=output_dir, csv_name=f'findg_{lambd:.2f}_{gamma:.2f}.csv',
            # What this does it to dynamically adapt the gamma parameter to be
            # 1 - the initial alpha (so if 0.05 of the edges are to be expected missing, gamma will be 0.95)
            kwargs_callback=lambda alpha: dict(
                lambd=lambd,
                gamma=np.clip(1 - gamma * alpha, 0., 1.)
            )
        )


if __name__ == '__main__':
    main()
