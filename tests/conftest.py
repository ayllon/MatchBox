import numpy as np
import pytest
from pandas import DataFrame

from matchbox.attributeset import AttributeSet
from matchbox.hypergraph import Graph, Edge
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


@pytest.fixture
def hyperclique48():
    """
    From Koeller 2002, Example 4.5
    """
    return Graph(
        V={1, 2, 3, 4, 5},
        E=set(map(Edge, [
            {1, 2, 3}, {1, 3, 4}, {1, 2, 4}, {1, 5, 2},
            {2, 3, 4}, {3, 4, 5}
        ]))
    )


@pytest.fixture
def graph2():
    return Graph(
        V={1, 2, 3, 4},
        E=set(map(Edge, [
            {1, 2}, {2, 3}, {3, 4}, {4, 1}, {2, 4}
        ]))
    )
