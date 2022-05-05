import os

import numpy as np
import pandas

STRTYPE_TO_TYPE = dict(
    integer=np.int,
    real=np.float
)


def parse_keel_file(path: str, ncols: int = None, nonames: bool = False):
    """
    Read a Keel datafile
    """
    assert not nonames
    with open(path, 'rt') as fd:
        column_names = []
        column_types = {}
        relation_name = os.path.basename(path)

        line = fd.readline().strip()
        while line != '@data':
            fields = line.split(maxsplit=1)
            if fields[0] == '@attribute':
                column_name, column_type = fields[1].split(maxsplit=1)
                column_type = column_type[:column_type.find('[')].strip()
                column_types[column_name] = STRTYPE_TO_TYPE[column_type]
            elif fields[0] in ['@inputs', '@outputs']:
                column_names.extend([f.strip() for f in fields[1].split(',')])

            line = fd.readline().strip()
        if not ncols:
            ncols = len(column_names)
        dataframe = pandas.read_csv(fd, delimiter=',', names=column_names[:ncols], dtype=column_types)
        dataframe.name = relation_name
        return dataframe
