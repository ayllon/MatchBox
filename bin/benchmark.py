#!/usr/bin/env python3
# coding: utf-8
import itertools
import logging
import os
import signal
import sys
import time
import uuid
import warnings
from argparse import ArgumentParser
from functools import reduce
from typing import List, Tuple, Set, Callable, Type, Iterable

import numpy as np
import pandas
from fasteners import InterProcessLock
from numpy.random import BitGenerator
from pandas import DataFrame

try:
    import matchbox
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '..'))

from matchbox.find2 import Find2
from matchbox.find_gamma import FindGamma
from matchbox.hypergraph import generate_graph
from matchbox.ind import Ind, unique_inds, find_max_arity_per_pair
from matchbox.tests import knn_test
from matchbox.util.callcounter import CallCounter
from matchbox.util.plot import to_dot_file
from matchbox.util.timing import Timing
from matchbox.util.loaders import load_datasets
from matchbox.uintersect import UIntersectFinder
from matchbox.gennext import gen_next

logger = logging.getLogger('benchmark')


def generate_uind(dataframes: List[Tuple[str, DataFrame]], alpha: float, output_dir: str, ncolumns: int) -> Set[Ind]:
    """
    Run the unary finder n the given dataframes

    Parameters
    ----------
    dataframes : list of dataframes
    alpha : float
        Significance level for the KS statistic
    output_dir : str
        Where to write the statistics to
    ncolumns : int
        Number of columns present

    Returns
    -------
    out : set of Ind
        Set of unary IND
    """

    uind_csv = os.path.join(output_dir, 'uind.csv')

    timing = Timing()
    uind_finder = UIntersectFinder(method='ks')
    for df_name, df in dataframes:
        uind_finder.add(df_name, df)
    with timing:
        uinds = uind_finder(alpha=alpha)

    logger.info('Number of UIND: %d', len(uinds))
    logger.info('Took %.2f seconds', timing.elapsed)

    uind_name_match = list(filter(lambda u: u.lhs.attr_names == u.rhs.attr_names, uinds))
    logger.info('%d unary IND found with matching names', len(uind_name_match))

    with InterProcessLock(uind_csv + '.lock'):
        results = pandas.DataFrame(
            {
                'columns': [ncolumns], 'uinds': [len(uinds)], 'match': [len(uind_name_match)],
                'time': [timing.elapsed], 'tests': uind_finder.ntests
            }
        )
    results.to_csv(uind_csv, mode='a', index=False, header=not os.path.exists(uind_csv))

    return uinds


def bootstrap_ind(uinds: Set[Ind], stop: int, alpha: float, test_method: Callable, test_args: dict, output_dir: str) \
        -> Tuple[Set[Ind], int]:
    """
    Generate and validate a set of IND of arity `stop` from a given initial set of unary IND.
    Intermediate validations are not done so the expected number of missing edges is easier to understand.

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
    output_dir : str
        Where to write the statistics to

    Returns
    -------
    out : Set of Ind
        Validated set of n-IND

    n : int
        Number of exact name matches
    """
    bootstrap_csv = os.path.join(output_dir, 'bootstrap.csv')

    timing = Timing()

    with timing:
        candidates = uinds
        for _ in range(1, stop):
            candidates = gen_next(candidates)

        logger.info('Number of candidate %d-IND: %d', stop, len(candidates))
        inds = set()
        for candidate in candidates:
            candidate.confidence = test_method(candidate.lhs.data, candidate.rhs.data, **test_args)
            if candidate.confidence >= alpha:
                inds.add(candidate)

    logger.info('Number of %d-IND: %d', stop, len(inds))

    induced_uind = list(reduce(frozenset.union, map(Ind.get_all_unary, inds), frozenset()))
    uind_name_match = list(filter(lambda u: u.lhs.attr_names == u.rhs.attr_names, induced_uind))
    logger.info('%d unary IND found with matching names after bootstrapping', len(uind_name_match))

    with InterProcessLock(bootstrap_csv + '.lock'):
        results = pandas.DataFrame(
            {
                'candidates': [len(candidates)], 'accepted': [len(inds)],
                'match': [len(uind_name_match)], 'time': [timing.elapsed]
            }
        )
        results.to_csv(bootstrap_csv, mode='a', index=False, header=not os.path.exists(bootstrap_csv))

    return inds, len(uind_name_match)


def _timeout_handler(signum: int, stack):
    raise TimeoutError()


def run_finder(Finder: Type, timeout: int, cross_datasets: List[Tuple[str, str]], ncolumns: int,
               alpha: float,
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
    timeout : int
        Timeout in seconds
    cross_datasets : List of dataset name pairs
        Needed to generate the output
    ncolumns : int
        Total number of columns
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
    signal.signal(signal.SIGALRM, _timeout_handler)

    results = {
        'id': [], 'columns': [], 'exact': [], 'bootstrap_alpha': [], 'time': [], 'tests': [], 'ind': [],
        'unique_ind': [], 'timeout': []
    }
    for cd in cross_datasets:
        results[f'max_{cd[0]}_{cd[1]}'] = list()

    finder_name = Finder.__name__

    for b_alpha in bootstrap_alphas:
        logger.info('%s starting with alpha=%.2f', finder_name, b_alpha)
        selected_ind = [ind for ind in bootstrap_ind if ind.confidence >= b_alpha]
        counted_test = CallCounter(test_method)

        finder = Finder(
            n=bootstrap_arity, alpha=alpha, test=counted_test, test_args=test_args,
            **kwargs_callback(b_alpha)
        )

        timing = Timing()
        timeout_flag = False
        with timing:
            try:
                if timeout:
                    signal.alarm(timeout)
                nind = finder(selected_ind)
                signal.alarm(0)
            except TimeoutError:
                logger.warning('Measurement timeout')
                nind = set()
                timeout_flag = True

        unique_nind = unique_inds(nind)

        run_id = str(uuid.uuid1())
        results['id'].append(run_id)
        results['columns'].append(ncolumns)
        results['exact'].append(exact)
        results['bootstrap_alpha'].append(b_alpha)
        results['time'].append(timing.elapsed)
        results['tests'].append(counted_test.counter)
        results['ind'].append(len(nind))
        results['unique_ind'].append(len(unique_nind))
        results['timeout'].append(timeout_flag)

        max_inds = find_max_arity_per_pair(unique_nind)
        for cd in cross_datasets:
            key = frozenset(cd)
            if key in max_inds:
                results[f'max_{cd[0]}_{cd[1]}'].append(max_inds[key].arity)
            else:
                results[f'max_{cd[0]}_{cd[1]}'].append(0)

        nind_dir = os.path.join(output_dir, run_id[:2], run_id)
        os.makedirs(nind_dir)

        # Save a summary of the arities
        arities = [i.arity for i in unique_nind]
        bins, counts = np.unique(arities, return_counts=True)
        np.savetxt(os.path.join(nind_dir, 'histogram.txt'), np.stack([bins, counts]).astype(np.int32), fmt='%d')

        # Save all ind
        with open(os.path.join(nind_dir, 'nind.txt'), 'wt') as fd:
            for i in unique_nind:
                print(f'{i.arity} {i.confidence:.2f} {i}', file=fd)

    result = pandas.DataFrame(results)
    csv_path = os.path.join(output_dir, csv_name)
    with InterProcessLock(csv_path + '.lock'):
        result.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))


def define_arguments() -> ArgumentParser:
    """
    Initialize an ArgumentParser
    """
    parser = ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='Debug logging')
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
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.05, 0.1],
                        help='Significance level for the Hyper-geometric test on the degrees of the nodes')
    parser.add_argument('--gammas', type=float, nargs='+', default=[1.],
                        help='Gamma factor for the number of missing edges on a quasi-clique')
    parser.add_argument('--columns', type=int, default=None,
                        help='Select a subset of the columns')
    parser.add_argument('--files', type=int, default=None,
                        help='Select a subset of the files, but reserve space on the output for the max arity found')
    parser.add_argument('--write-dot', action='store_true',
                        help='Write a dot file with the initial graph')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Repeat the test these number of times')
    parser.add_argument('--no-find2', action='store_true',
                        help='Do not run Find2')
    parser.add_argument('--no-grow', action='store_true',
                        help='Do not run with growing stage')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds')
    parser.add_argument('--no-column-names', action='store_true', help='The dataset has no column names')
    parser.add_argument('data', metavar='DATA', nargs='+', help='Dataset')
    return parser


def main():
    """
    Entry point
    """
    # Parse arguments
    parser = define_arguments()
    args = parser.parse_args()

    # Basic setup
    log_level = logging.INFO if not args.debug else logging.DEBUG
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.basicConfig(format='%(asctime)s %(name)15.15s %(levelname)s\t%(message)s',
                        level=log_level, stream=sys.stderr)
    logging.getLogger('filelock').setLevel(logging.WARN)

    # Initialize the random state
    random_generator = np.random.MT19937(args.seed)

    # Fill defaults
    if not args.id:
        args.id = '_'.join(map(os.path.basename, args.data))
    output_dir = os.path.join(args.output_dir, args.id)
    logger.info(f'Using output directory {output_dir}')

    # Create output directory if necessary
    os.makedirs(output_dir, exist_ok=True)
    sampling_csv = os.path.join(output_dir, 'sampling.csv')

    # Common statistical test setup
    test_args = dict(k=args.k, n_perm=args.permutations)
    test_method = knn_test

    # Load datasets
    datasets = load_datasets(args.data, ncols=args.columns, nonames=args.no_column_names, nframes=args.files)

    # Dataset combinations, required to be deterministic between runs regardless of the result
    dataset_names = [d[0] for d in datasets]
    cross_datasets = list(map(tuple, itertools.combinations(dataset_names, 2)))
    ncolumns = sum([len(d[1].columns) for d in datasets])

    # We draw different samples each time, so we need to restart from the beginning,
    # but at least we avoid re-loading the datasets
    for i in range(1, args.repeat + 1):
        logger.info('Iteration %d / %d', i, args.repeat)

        # Get samples
        timing = Timing()
        with timing:
            samples = [
                (name, df.sample(args.sample_size, replace=True, random_state=random_generator)) for name, df in
                datasets if len(df) > 0
            ]

        with InterProcessLock(sampling_csv + '.lock'):
            pandas.DataFrame({'columns': [ncolumns], 'time': [timing.elapsed]}) \
                .to_csv(sampling_csv, mode='a', index=False, header=not os.path.exists(sampling_csv))

        # Initial set of unary IND
        logger.info('Looking for unary IND')
        uinds = generate_uind(samples, alpha=args.uind_alpha, output_dir=output_dir, ncolumns=ncolumns)

        # Bootstrap initial set of IND using mind
        # The individual methods can bootstrap themselves, but we want to have the initial conditions
        # as close as possible
        initial_ind, exact = bootstrap_ind(
            uinds, stop=args.bootstrap_arity, alpha=min(args.bootstrap_alpha),
            test_method=test_method, test_args=test_args, output_dir=output_dir
        )

        # Write dot file with the initial graph
        if args.write_dot:
            graph_dot_file = os.path.join(output_dir, f'{args.bootstrap_arity}-graph.dot')
            bootstrap_graph, _ = generate_graph(initial_ind)
            to_dot_file(bootstrap_graph, graph_dot_file)
            logger.info('Initial graph written to %s', graph_dot_file)

        # Benchmark find2
        if not args.no_find2:
            run_finder(
                Find2, timeout=args.timeout, cross_datasets=cross_datasets, ncolumns=ncolumns, alpha=args.nind_alpha,
                bootstrap_arity=args.bootstrap_arity, bootstrap_ind=initial_ind, bootstrap_alphas=args.bootstrap_alpha,
                exact=exact, test_method=test_method, test_args=test_args,
                output_dir=output_dir, csv_name='find2.csv'
            )
        else:
            logger.warning('Skipping Find2 run')

        # Benchmark findg
        grow_flags = [False]
        if not args.no_grow:
            grow_flags.append(True)

        for lambd, gamma, grow in itertools.product(args.lambdas, args.gammas, grow_flags):
            # This combination is too lax, anything would be accepted
            if lambd <= 0 and gamma >= 100:
                continue
            logger.info('FindG lambda=%.2f, gamma=1 - %.2f * alpha, grow=%d', lambd, gamma, grow)
            run_finder(
                FindGamma, timeout=args.timeout, ncolumns=ncolumns, cross_datasets=cross_datasets,
                alpha=args.nind_alpha,
                bootstrap_arity=args.bootstrap_arity, bootstrap_ind=initial_ind, bootstrap_alphas=args.bootstrap_alpha,
                exact=exact, test_method=test_method, test_args=test_args,
                output_dir=output_dir, csv_name=f'findg_{lambd:.2f}_{gamma:.2f}_{grow:d}.csv',
                # What this does it to dynamically adapt the gamma parameter to be
                # 1 - the initial alpha (so if 0.05 of the edges are to be expected missing, gamma will be 0.95)
                kwargs_callback=lambda alpha: dict(
                    lambd=lambd,
                    gamma=np.clip(1 - gamma * alpha, 0., 1.),
                    grow=grow,
                )
            )


if __name__ == '__main__':
    main()
