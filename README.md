MatchBox
========

This repository contains the [implementation in Python](matchbox/) of PresQ, an algorithm
aimed at infering common multidimensional ``equally distributed'' attributes on numerical data
with associated uncertainties.

Is is based on on Find2[^1], but it accounts for the the effect that statistical errors can have (i.e. falsely
reject or accept that two multidimensional set of attributes are equally distributed) by
performing an approximate search of *quasi-cliques* on uniform hypergraphs.

The paper/s is/are in progress.

It also includes an implementation of Mind[^2] to bootstrap both Find2 and PresQ.

## Benchmarks

[`bin/benchmark.py`](bin/benchmark.py) contains an script that can be used to benchmark Find2 and PresQ.
It supports [FITS files](https://en.wikipedia.org/wiki/FITS) - thanks to [Astropy](https://docs.astropy.org/) -,
CSV files - thanks to [Pandas](https://pandas.pydata.org/) -, and the [Keel format](https://sci2s.ugr.es/keel/datasets.php).

Benchmarks are relatively easy to run

```console
usage: benchmark.py [-h] [--id ID] [--output-dir OUTPUT_DIR] [--seed SEED] [--sample-size SAMPLE_SIZE] [-k K] [-p PERMUTATIONS] [--uind-alpha UIND_ALPHA] [--nind-alpha NIND_ALPHA]
                    [--bootstrap-alpha BOOTSTRAP_ALPHA [BOOTSTRAP_ALPHA ...]] [--bootstrap-arity BOOTSTRAP_ARITY] [--lambdas LAMBDAS [LAMBDAS ...]] [--gammas GAMMAS [GAMMAS ...]]
                    [--columns COLUMNS] [--write-dot] [--repeat REPEAT]
                    DATA1 DATA2

positional arguments:
  DATA1                 Dataset 1
  DATA2                 Dataset 2

optional arguments:
  -h, --help            show this help message and exit
  --id ID               Run identifier, defaults to a derived from the dataset file names
  --output-dir OUTPUT_DIR
                        Write the generated output to this directory
  --seed SEED           Initial random seed
  --sample-size SAMPLE_SIZE
                        Sample size
  -k K                  Number of neighbors for the KNN test
  -p PERMUTATIONS, --permutations PERMUTATIONS
                        Number of permutations for the KNN test
  --uind-alpha UIND_ALPHA
                        Significance level for the unary IND tests (KS)
  --nind-alpha NIND_ALPHA
                        Significance level for the n-IND tests (KNN)
  --bootstrap-alpha BOOTSTRAP_ALPHA [BOOTSTRAP_ALPHA ...]
                        Significance levels for the bootstrapping tests (KNN)
  --bootstrap-arity BOOTSTRAP_ARITY
                        Run MIND up to this arity
  --lambdas LAMBDAS [LAMBDAS ...]
                        Significance level for the Hyper-geometric test on the degrees of the nodes
  --gammas GAMMAS [GAMMAS ...]
                        Gamma factor for the number of missing edges on a quasi-clique
  --columns COLUMNS     Select a subset of the columns
  --write-dot           Write a dot file with the initial graph
  --repeat REPEAT       Repeat the test these number of times
```

### Setup

For convenience, [`environment.yml`](environment.yml) contains all that is necessary to create
a [conda](https://docs.conda.io/) environment with the required dependencies, both for running
the benchmark, but also to (re-)run the notebooks.

### Datasets
References to the used datasets are contained under [`data/`](/data/), although the files themselves
are not included. In any case they are open and easy to obtain.

### Notebooks
Under [`notebooks/`](notebooks/) there are notebooks and some statistics about past runs.
These notebooks should be easy to re-use with the output of `bin/benchmark.py` compressed as `tar.gz` files.


[^1]: A. Koeller and E. A. Rundensteiner, “Discovery of high-dimensional inclusion dependencies,” in Proceedings 19th international conference on data engineering (cat. No. 03CH37405), 2003, pp. 683–685,
doi: [10.1109/ICDE.2003.1260834](https://doi.org/10.1109/ICDE.2003.1260834).

[^2]: F. De Marchi, S. Lopes, and J.-M. Petit, “Efficient algorithms for mining inclusion dependencies,” in International conference on extending database technology, 2002, pp. 464–476,
doi: [10.1007/3-540-45876-X_30](https://doi.org/10.1007/3-540-45876-X_30)
