import logging
import sqlite3
from typing import List, Tuple

import numpy as np
import pandas
from numpy.random import BitGenerator

from matchbox.util.timing import Timing

logger = logging.getLogger(__name__)


class DbAdapter:
    """
    Wrap a sqlite connection and table name in an API compatible with the expectations of benchmark.py
    """

    class ColumnWrapper:
        def __init__(self, colum: str, sqlite: sqlite3.Connection, table: str):
            self.__column = colum
            self.__sqlite = sqlite
            self.__table = table

        def unique(self) -> List:
            return self.__sqlite.execute(f'SELECT DISTINCT {self.__column} FROM {self.__table} LIMIT 11').fetchall()

    def __init__(self, sqlite: sqlite3.Connection, table: str, ncols: int):
        self.__sqlite = sqlite
        self.__table = table
        self.__dropna = [None, None]
        self.__columns = []
        for col_info in sqlite.execute(f'PRAGMA table_info({self.__table})'):
            if ncols and len(self.__columns) == ncols:
                break
            self.__columns.append(col_info[1])
        self.__len = sqlite.execute(f'SELECT COUNT(*) FROM {self.__table}').fetchone()[0]
        # Let Pandas figure out the types
        df = pandas.read_sql_query(f'SELECT {",".join(self.__columns)} FROM {self.__table} LIMIT 1',
                                   self.__sqlite)
        self.__dtypes = df.dtypes

    def dropna(self, axis: int, how: str, inplace: bool):
        assert inplace
        self.__dropna[axis] = how

    @property
    def columns(self):
        return self.__columns

    @property
    def dtypes(self):
        return self.__dtypes

    def __len__(self):
        return self.__len

    def __getitem__(self, item: str):
        return DbAdapter.ColumnWrapper(item, self.__sqlite, self.__table)

    def __delitem__(self, key):
        self.__columns.remove(key)

    def sample(self, sample_size: int, replace: bool, random_state: BitGenerator):
        logger.debug('Sampling %d from %s', sample_size, self.__table)
        timing = Timing()
        with timing:
            cutout = -9223372036854775808 + np.ceil((sample_size * 1.5 / self.__len) * 18446744073709551615)
            df = pandas.read_sql_query(f'SELECT {",".join(self.__columns)} FROM {self.__table} WHERE RANDOM() <= ?',
                                       self.__sqlite, params=(cutout,))
        logger.debug('Done in %.2f seconds (%d)', timing.elapsed, len(df))
        if self.__dropna[0]:
            df.dropna(axis=0, how=self.__dropna[0], inplace=True)
        if self.__dropna[1]:
            df.dropna(axis=1, how=self.__dropna[1], inplace=True)
        return df.sample(sample_size, replace=replace, random_state=random_state)


def load_sqlite(path: str, ncols: int, nonames: bool) -> List[Tuple[str, DbAdapter]]:
    """
    Load multiple datasets from within a sqlite database
    """
    assert not nonames
    conn = sqlite3.connect(path)
    tables = []
    for table in conn.execute('SELECT name from sqlite_master where type= "table"'):
        tables.append((table[0], DbAdapter(conn, table[0], ncols)))
    logger.info('Found %d tables in %s', len(tables), path)
    return tables
