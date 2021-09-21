#!/usr/bin/env python3
# coding: utf-8
import itertools
import logging
import os
import signal
import sys
import time
import warnings
from argparse import ArgumentParser
from typing import Tuple, FrozenSet, Callable, Union, List

import numpy as np
import pandas
from filelock import FileLock

try:
    import matchbox
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '..'))

from matchbox.find2 import find_hypercliques
from matchbox.find_gamma import find_quasicliques
from matchbox.hypergraph import Graph, Edge

try:
    import matchbox
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '..'))

logger = logging.getLogger('benchmark-cuasi')


def generate_clique(rank: int, cardinality: int, additional: Union[int, float]) -> Graph:
    """
    Generate a graph with a full clique and a set of additional, disconnected nodes

    Parameters
    ----------
    rank : int
        Rank of the graph
    cardinality : int
        Cardinality of the quasi clique
    additional : int
        Additional, disconnected, nodes
    """
    if additional < 1:
        additional = additional * cardinality
    additional = int(additional)
    clique = list(range(cardinality))
    extra = list(range(len(clique), additional + len(clique)))
    edges = map(Edge, itertools.combinations(clique, rank))
    return Graph(clique + extra, edges), frozenset(clique)


def disturb_graph(G_seed: Graph, missing_edges: float, extra_edges: float,
                  rng: np.random.BitGenerator) -> Tuple[Graph, int, int]:
    generator = np.random.Generator(rng)
    # Generate all possible edges
    candidate_edges = set(map(Edge, itertools.combinations(G_seed.V, G_seed.rank)))
    # Remove those already on the clique
    candidate_edges.difference_update(G_seed.E)
    # Pick random set from the original clique
    remove_n = int(len(G_seed.E) * missing_edges)
    pick_n = len(G_seed.E) - remove_n
    logger.info('Dropping %d edges from the clique', remove_n)
    clique_e = generator.choice(list(G_seed.E), size=pick_n, replace=False)
    # Pick random set from the additional set
    add_n = int(len(candidate_edges) * extra_edges)
    logger.info('Adding %d edges', add_n)
    add_e = generator.choice(list(candidate_edges), size=add_n, replace=False)
    # Build graph
    return Graph(G_seed.V, set(clique_e).union(add_e)), remove_n, add_n


def _timeout_handler(signum: int, stack):
    raise TimeoutError()


def measure(clique: FrozenSet, G: Graph, finder: Callable[..., FrozenSet[Edge]], /, timeout: int = None,
            **kwargs) -> Tuple[float, float]:
    """
    Measure run time and recovery ratio for the given finder

    Parameters
    ----------
    clique : FrozenSet
        Target clique
    G : Graph
        Input graph
    finder : Callable
        Finder implementation
    timeout: int
        Stop the execution after this many seconds
    kwargs : dict
        Forwarded to finder

    Returns
    -------
    out : tuple(float, float)
        Elapsed time (in seconds), relative size of the found clique wrt the parameter clique
    """
    logger.info('Measuring %s', finder.__name__)
    if timeout:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    start = time.perf_counter()
    try:
        quasi_cliques = finder(G, **kwargs)
        signal.alarm(0)
    except (TimeoutError, TypeError):
        logger.warning('Measurement timeout')
        return np.nan, np.nan

    end = time.perf_counter()
    elapsed = end - start

    max_ratio = 0.
    for q in quasi_cliques:
        if q.issuperset(clique) or clique.issuperset(q):
            ratio = len(q) / len(clique)
            if ratio > max_ratio:
                max_ratio = ratio

    return elapsed, max_ratio


def define_arguments() -> ArgumentParser:
    """
    Initialize an ArgumentParser
    """
    parser = ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='Debug logging')
    parser.add_argument('--seed', type=int, default=time.time_ns(),
                        help='Initial random seed')
    parser.add_argument('-k', '--rank', type=int, default=2, help='Hyper-graph rank')
    parser.add_argument('-n', '--cardinality', type=str, nargs='+', required=True,
                        help='Number of nodes on the quasi-clique')
    parser.add_argument('--repeat', type=str, nargs='+', default=['1'], help='Repeat with different initial conditions')
    parser.add_argument('-N', '--additional', type=float, default=0., help='Number of additional nodes')
    parser.add_argument('-a', '--missing-edges', type=float, default=0.05, help='Ratio of missing edges')
    parser.add_argument('-b', '--extra-edges', type=float, default=0.1, help='Ratio of additional edges')
    parser.add_argument('--dot', type=str, help='Save the generated graph into a dot file')
    parser.add_argument('-L', '--Lambda', type=float, default=0.05, help='Lambda parameter for FindQ')
    parser.add_argument('--timeout', type=int, default=60 * 60, help='Timeout in seconds')
    parser.add_argument('-o', '--output', type=str, metavar='CSV', default=None, help='Output CSV')
    return parser


def parse_interval(interval: List[str]) -> List[int]:
    out = []
    for i in interval:
        if ':' in i:
            start, stop = map(int, i.split(':'))
            if start < stop:
                out.extend(range(start, stop + 1))
            else:
                out.extend(reversed(range(stop, start + 1)))
        else:
            out.append(int(i))
    return out


# noinspection PyPep8Naming
def main():
    """
    Entry point
    """
    # Parse arguments
    parser = define_arguments()
    args = parser.parse_args()

    # Convert intervals
    args.cardinality = parse_interval(args.cardinality)
    args.repeat = parse_interval(args.repeat)
    if len(args.repeat) == 1:
        args.repeat = [args.repeat[0]] * len(args.cardinality)

    if len(args.cardinality) != len(args.repeat):
        parser.error('--repeat must have one value, or as many as --cardinality')

    # Basic setup
    log_level = logging.INFO if not args.debug else logging.DEBUG
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.basicConfig(format='%(asctime)s %(name)15.15s %(levelname)s\t%(message)s',
                        level=log_level, stream=sys.stderr)

    # Initialize the random state
    random_generator = np.random.MT19937(args.seed)

    # Output
    df = dict(
        rank=[], clique=[], order=[], size=[], missing=[], added=[],
        find2_time=[], find2_ratio=[],
        findq_time=[], findq_ratio=[],
        findqg_time=[], findqg_ratio=[],
    )

    # For each pass
    for card, repeat in zip(args.cardinality, args.repeat):
        # Clear output
        for k in df:
            df[k] = list()

        logger.info('Running for a clique of size %d', card)
        # Generate the initial set (only full clique)
        G_seed, clique = generate_clique(args.rank, card, args.additional)
        logger.info('%d nodes, %d edges', len(G_seed.V), len(G_seed.E))
        for i in range(repeat):
            # Disturb the original graph
            G, removed, added = disturb_graph(G_seed, args.missing_edges, args.extra_edges, random_generator)

            # Add setup
            df['rank'].append(args.rank)
            df['clique'].append(card)
            df['order'].append(len(G.V))
            df['size'].append(len(G.E))
            df['missing'].append(args.missing_edges)
            df['added'].append(args.extra_edges)

            # Measure Find2
            f2_time, f2_ratio = measure(clique, G, find_hypercliques, timeout=args.timeout)

            # Measure FindQ
            fq_time, fq_ratio = measure(
                clique, G, find_quasicliques, timeout=args.timeout,
                lambd=args.Lambda, gamma=1 - args.missing_edges, grow=False)

            # Measure FindQ with growing stage
            fqg_time, fqg_ratio = measure(
                clique, G, find_quasicliques, timeout=args.timeout,
                lambd=args.Lambda, gamma=1 - args.missing_edges, grow=True)

            # Add result
            df['find2_time'].append(f2_time)
            df['find2_ratio'].append(f2_ratio)
            df['findq_time'].append(fq_time)
            df['findq_ratio'].append(fq_ratio)
            df['findqg_time'].append(fqg_time)
            df['findqg_ratio'].append(fqg_ratio)

        # Write
        if args.output:
            with FileLock(args.output + '.lock'):
                pandas.DataFrame(df).to_csv(args.output, mode='a', header=not os.path.exists(args.output))
        else:
            pandas.DataFrame(df).to_csv(sys.stdout, mode='a', header=True)


if __name__ == '__main__':
    main()
