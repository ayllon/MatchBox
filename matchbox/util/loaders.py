import logging
import os
from typing import List, Tuple

import numpy as np
import pandas
from pandas import DataFrame

from matchbox.util.keel import parse_keel_file

logger = logging.getLogger(__name__)


def load_fits(path: str, ncols: int) -> DataFrame:
    """
    Load a FITS catalog. Only the first table HDU is supported.
    """
    from astropy.table import Table
    df = Table.read(path).to_pandas()
    if ncols:
        df = df[df.columns[:ncols]]
    # Need to mask "special" -99 values
    df.replace(-99, np.nan, inplace=True)
    df.replace(-99.9, np.nan, inplace=True)
    return df


def load_csv(path: str, ncols: int) -> DataFrame:
    """
    Load a CSV file using pandas
    """
    cols = range(ncols) if ncols else None
    return pandas.read_csv(path, usecols=cols, skipinitialspace=True)


def load_tsv(path: str, ncols: int) -> DataFrame:
    """
    Load a TSV file u sing pandas
    """
    cols = range(ncols) if ncols else None
    return pandas.read_csv(path, usecols=cols, sep='\t', skipinitialspace=True)


# List of supported formats and their loaders
_loaders = {
    '.fits': load_fits,
    '.dat': parse_keel_file,
    '.csv': load_csv,
    '.data': load_csv,
    '.tsv': load_tsv,
}


def unambiguous_names(paths: List[str], nparents: int = 0) -> List[str]:
    """
    Given a list of dataset paths, return a list of unambiguous identifiers.
    For instance if two datasets share the filename, the parent path will be used to disambiguate.

    Parameters
    ----------
    paths : list of strings
        Paths to the datasets
    nparents : int
        Number of parents to use to disambiguate

    Returns
    -------
    out : List of unambiguous names
    """
    names = list()
    for path in paths:
        names.append(os.path.join(*path.split(os.sep)[-nparents - 1:]))

    if len(set(names)) == len(paths):
        return names
    return unambiguous_names(paths, nparents + 1)


def load_datasets(paths: List[str], ncols: int = None, filter_nan: str = 'column') -> List[Tuple[str, DataFrame]]:
    """
    Load a list of datasets from files

    Parameters
    ----------
    paths : list of strings
        Paths to the datasets
    ncols : int
        Read only this number of columns (all if 0 or None)
    filter_nan : str
        Remove NaN columns and rows, or both

    Returns
    -------
    out : List of tuples (name, dataframe)
    """
    dataframes = []
    names = unambiguous_names(paths)
    for path, name in zip(paths, names):
        logger.info('Loading %s', path)
        ext = os.path.splitext(path)[1]
        df = _loaders[ext](path, ncols)
        if filter_nan:
            with pandas.option_context('mode.use_inf_as_na', True):
                if filter_nan in ['column', 'both']:
                    df.dropna(axis=1, how='all', inplace=True)
                if filter_nan in ['row', 'both']:
                    df.dropna(axis=0, how='any', inplace=True)
        logger.info('\t%d columns loaded', len(df.columns))
        dataframes.append((name, df))
    return dataframes
