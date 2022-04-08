import logging
from typing import List

import numpy as np
import pandas

logger = logging.getLogger(__name__)

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


def prune_columns(table: pandas.DataFrame, dtypes: List = None):
    """
    Drop columns with all NaN, constant and multidimensional
    """
    if dtypes is None:
        dtypes = _types_reverse.keys()

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
        elif table[c].dtype.type not in dtypes:
            logger.debug('Drop %s because it has a filtered type', c)
            del table[c]


def group_columns_by_type(table: pandas.DataFrame):
    """
    Create a dictionary where the key is the string representation of the column data type (i.e. 'int' or 'string')
    and the value is the list of columns with such data type
    """
    columns_group = dict([(k, []) for k in _types.keys()])
    for c in table.columns:
        columns_group[_types_reverse[table[c].dtype.type]].append(c)
    return columns_group
