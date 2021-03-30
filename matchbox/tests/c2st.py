import numpy as np
import pandas
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def c2s_test(lhs_data: pandas.DataFrame, rhs_data: pandas.DataFrame, classifier='nn') -> float:
    """
    Based on
    "Revisiting Classifier Two-Sample Tests", Lopez 2016
    See page 3

    Parameters
    ----------
    lhs_data : pandas.DataFrame
        LHS data set
    rhs_data : pandas.DataFrame
        RHS data set
    classifier : str or object
        'nn' for Neural Network, 'knn' for nearest neighbor, or a classifier instance
    Returns
    -------
    p-value : float
    """
    if isinstance(classifier, str):
        if classifier == 'knn':
            classifier = KNeighborsClassifier()
        elif classifier == 'nn':
            classifier = MLPClassifier()
        else:
            raise ValueError(f'Unknown classifier {classifier}')
    assert hasattr(classifier, 'fit')
    assert hasattr(classifier, 'predict')

    # First: Construct the dataset
    if isinstance(lhs_data, pandas.DataFrame):
        lhs_data = lhs_data.to_numpy()
    if isinstance(rhs_data, pandas.DataFrame):
        rhs_data = rhs_data.to_numpy()
    X = np.concatenate([lhs_data, rhs_data])
    Y = np.concatenate([np.zeros(len(lhs_data)), np.ones(len(rhs_data))])

    # Second: Shuffle and split into train/test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True)

    # Third: Train
    classifier.fit(X_train, Y_train)

    # Fourth: Test statistic is accuracy
    score = classifier.score(X_test, Y_test)

    # Under H0, the score can be approximated by N(1/2, 1/(4|X_test|))
    mean, std = 0.5, np.sqrt(1 / (4 * len(X_test)))
    p = norm.cdf(score, mean, std)
    if p > 0.5:
        return 2 * (1 - p)
    return 2 * p
