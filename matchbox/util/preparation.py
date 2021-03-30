import logging
from typing import Callable

import numpy as np
import pandas

logger = logging.getLogger(__name__)


def prune_columns(table: pandas.DataFrame, nan_replace: Callable[[str, np.array], np.array] = None,
                  nan_threshold: float = 1.):
    """
    Drop columns with all NaN, constant and multidimensional
    """
    for c in table.columns:
        if hasattr(table[c], 'count') and table[c].count() == 0:
            logger.debug('Drop %s because all entries are masked', c)
            del table[c]
        elif table[c].dtype.type in (np.float, np.float32, np.float64):
            if len(table[c].shape) > 1:
                logger.debug('Drop %s because multidimensional columns are not supported', c)
                del table[c]
            elif np.nanstd(table[c]) == 0:
                logger.debug('Drop %s because it is constant', c)
                del table[c]
            else:
                if nan_replace:
                    table.loc[nan_replace(c, table[c]), c] = np.nan

                try:
                    if ~np.isfinite(table[c]).sum() / len(table) >= nan_threshold:
                        logger.debug('Drop %s because at least %f values are NaN', c, nan_threshold)
                        del table[c]
                except ValueError as e:
                    logger.debug('Drop %s (%s)', c, str(e))
                    del table[c]
    table.dropna(axis=1, how='all', inplace=True)


_types = {
    'string': [str, np.bytes_],
    'float': [float, np.float32, np.float64],
    'int': [int, np.int16, np.int32, np.int64],
    'uint': [np.uint16, np.uint32, np.uint64],
    'object': [np.object_]
}
_types_reverse = dict()
for k, vs in _types.items():
    for v in vs:
        _types_reverse[v] = k


def group_columns_by_type(table: pandas.DataFrame):
    """
    Create a dictionary where the key is the string representation of the column data type (i.e. 'int' or 'string')
    and the value is the list of columns with such data type
    """
    columns_group = dict([(k, []) for k in _types.keys()])
    for c in table.columns:
        columns_group[_types_reverse[table[c].dtype.type]].append(c)
    return columns_group
