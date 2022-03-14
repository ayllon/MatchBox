from typing import Tuple, Union

import numpy as np
import pandas
import somoclu
from scipy.stats import chi2


def som_test(lhs_data: pandas.DataFrame, rhs_data: pandas.DataFrame, size: Tuple[int, int] = (10, 10),
             ret_som: bool = False,
             ret_counts: bool = False, **kwargs) -> Union[float, Tuple[float, ...]]:
    # To numpy arrays
    if isinstance(lhs_data, pandas.DataFrame):
        lhs_data = lhs_data.dropna(axis=0, how='any').to_numpy()
    if isinstance(rhs_data, pandas.DataFrame):
        rhs_data = rhs_data.dropna(axis=0, how='any').to_numpy()

    c = np.concatenate([lhs_data, rhs_data]).astype(np.float32)
    som = somoclu.Somoclu(size[0], size[1], **kwargs)
    som.train(c)

    # https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm
    ap = som.get_surface_state(lhs_data)
    bp = som.get_surface_state(rhs_data)

    ap = (ap <= ap.min(axis=1)[:, np.newaxis]).sum(axis=0).reshape(size[1], size[0])
    bp = (bp <= bp.min(axis=1)[:, np.newaxis]).sum(axis=0).reshape(size[1], size[0])
    gt0 = (ap + bp) > 0
    c2 = np.sum(((ap - bp) ** 2)[gt0] / (ap + bp)[gt0])

    c = int(len(lhs_data) == len(rhs_data))
    ret = [1 - chi2.cdf(c2, df=gt0.sum() - c)]
    if ret_som:
        ret.append(som)
    if ret_counts:
        ret.append((ap, bp))
    if len(ret) > 1:
        return tuple(ret)
    return ret[0]
