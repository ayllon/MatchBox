import numpy as np
import pytest
from pandas import DataFrame

from matchbox.attributeset import AttributeSet
from matchbox.ind import Ind


@pytest.fixture
def table1():
    df = DataFrame(
        data={
            'X': np.linspace(0, 100, num=100),
            'Y': np.linspace(80, 180, num=100),
            'V': np.random.normal(0, 10, size=100)
        }
    )
    df['V'][np.argmin(df['V'])] = 0.
    df['V'][np.argmax(df['V'])] = 10.
    return df


@pytest.fixture
def table2():
    df = DataFrame(
        data={
            'X': np.linspace(0, 100, num=100),
            'Y': np.linspace(80, 180, num=100),
            'V': np.random.normal(5, 1, size=100)
        }
    )
    df['V'][np.argmin(df['V'])] = 0.
    df['V'][np.argmax(df['V'])] = 10.
    return df


@pytest.fixture
def table3():
    df = DataFrame(
        data={
            'X': np.linspace(101, 200, num=100),
            'Y': np.linspace(80, 180, num=100),
            'V': np.random.normal(0, 10, size=100)
        }
    )
    df['V'][np.argmin(df['V'])] = 0.
    df['V'][np.argmax(df['V'])] = 10.
    return df

@pytest.fixture
def table4():
    df = DataFrame(
        data={
            'A': np.linspace(101, 200, num=100),
            'B': np.linspace(80, 180, num=100),
            'C': np.random.normal(0, 10, size=100)
        }
    )
    return df

@pytest.fixture
def ind():
    return Ind(
        lhs=AttributeSet('R', ['A']),
        rhs=AttributeSet('S', ['E'])
    )


@pytest.fixture
def ind2():
    return Ind(
        lhs=AttributeSet('R', ['A', 'B']),
        rhs=AttributeSet('S', ['E', 'F'])
    )


@pytest.fixture
def ind3():
    return Ind(
        lhs=AttributeSet('R', ['A', 'B', 'C']),
        rhs=AttributeSet('S', ['E', 'F', 'G'])
    )
