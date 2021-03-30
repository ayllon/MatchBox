import logging
from typing import Callable, Iterable, List, Dict

import numpy as np
import pandas
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from matchbox.util import longest_prefix

_logger = logging.getLogger(__name__)


def generate_schema_hash(df: pandas.DataFrame) -> int:
    """
    Returns
    -------
    out : int
        A hash from the column hashes and types
    """
    h = 0
    for name, dtype in dict(df.dtypes).items():
        h ^= hash(name) ^ hash(dtype)
    return h


def separable(data_frames: Iterable[pandas.DataFrame], samples: int = 1000, threshold: float = 0.8,
              use_rfecv: bool = True, rfecv_args: dict = None,
              scorer: Callable[[np.ndarray, np.ndarray], float] = accuracy_score) -> (bool, object):
    """
    Trains a classifier to learn to "guess" the files where a given set of records are.

    Parameters
    ----------
    data_frames : iterable of pandas.DataFrame
        One data frame per file
    samples : int
        How many samples per file to take
    threshold : float
        Files will be consider separable if the model score on the test set is greater or equal to this
        threshold
    use_rfecv : bool
        Use recursive feature elimination with cross-validation
    rfecv_args : dict
        Parameters to pass down to the RFECV
    scorer : Callable
        A callable that receives the true label and the prediction label, and returns a float
        scoring the result
    Returns
    -------
    result, model : bool, object
        True if the model has been able to tell the files apart, and the model itself
    """
    if not rfecv_args:
        rfecv_args = dict(cv = 5)
    if samples:
        _logger.debug(f'Generating {samples} samples per data frame ({samples * len(data_frames)})')
        all_samples = pandas.concat([df.sample(samples, replace=True) for df in data_frames]).dropna(axis=1, how='all')
        feat_cols = all_samples.columns
        all_samples['FILE'] = np.repeat(np.arange(len(data_frames)), samples)
    else:
        all_samples = pandas.concat(data_frames).dropna(axis=1, how='all')
        files = []
        for i, df in enumerate(data_frames):
            files.append(np.full(len(df), i))
        feat_cols = all_samples.columns
        all_samples['FILE'] = np.concatenate(files)

    _logger.debug(f'{len(feat_cols)} raw features available')

    _logger.debug('Filtering NaN')
    filtered = all_samples.dropna(axis=0)
    _logger.debug(f'{len(filtered)} instances after NaN filtering')

    _logger.debug('Train/test split')
    train, test = train_test_split(filtered)

    classifier = DecisionTreeClassifier()
    _logger.debug(f'Using {classifier} for classifying the objects...')

    if use_rfecv:
        classifier = RFECV(estimator=classifier, **rfecv_args)
        _logger.debug(f'Using Recursive Feature Elimination with Cross-Validation')
    classifier.fit(train[feat_cols], train['FILE'])
    _logger.debug(f'Classifier trained!')
    if use_rfecv:
        _logger.debug(f'{classifier.n_features_} features selected.')

    y_train_pred = classifier.predict(train[feat_cols])
    train_score = scorer(train['FILE'], y_train_pred)
    _logger.info(f'Score of {train_score} on the train set')

    _logger.debug('Performing prediction on the test set...')

    y_test_pred = classifier.predict(test[feat_cols])
    test_score = scorer(test['FILE'], y_test_pred)
    _logger.info(f'Score of {test_score:.3f} on the test set, threshold set at {threshold}')
    setattr(classifier, 'feature_names_', np.array(list(map(str, feat_cols))))
    return test_score >= threshold, classifier


class DataframeRegistry(object):
    """
    Store a set of data frames
    """

    def __init__(self):
        self.__dataframes = {}
        self.__schema = {}
        self.supersets_ = {}
        self.classifiers_ = {}

    def add(self, name: str, df: pandas.DataFrame):
        """
        Register a new dataframe
        """
        if name in self.__dataframes:
            raise KeyError(f'{name} already registered')
        self.__dataframes[name] = df
        schema_hash = generate_schema_hash(df)
        if schema_hash not in self.__schema:
            self.__schema[schema_hash] = []
        self.__schema[schema_hash].append(name)
        _logger.debug(f'Registered {name} with hash {schema_hash} ({len(self.__schema[schema_hash])} hits)')

    def __len__(self) -> int:
        return len(self.__dataframes)

    def __get_frames(self, frame_names) -> list:
        return list([self.__dataframes[n] for n in frame_names])

    @property
    def candidate_sets_(self) -> List[List[str]]:
        """
        Returns
        -------
        out : list of list strings
            Each position on the list corresponds to a list of files with shared schema
        """
        return list(self.__schema.values())

    def supersets(self, **kwargs) -> Dict[str, pandas.DataFrame]:
        """
        Find partitioned data frames

        Parameters
        ----------
        kwargs : dict
            Arguments to pass down to `separable`

        Returns
        -------
        out : dictionary of pandas data frame
            The key corresponds to the group name (longest common prefix), and the value to a
            merged data frame
        """
        if self.supersets_:
            _logger.debug('Supersets already computed')
            return self.supersets_
        for schema_hash, frame_names in self.__schema.items():
            if len(frame_names) == 1:
                _logger.debug(f'Single match for {frame_names}')
                self.supersets_[frame_names[0]] = self.__dataframes[frame_names[0]]
                continue
            # Readable candidate group name
            group_name = longest_prefix(frame_names)
            if not group_name:
                group_name = schema_hash
            # Try learning the separation between files
            _logger.info(f'Checking separability of group {group_name}')
            are_separable, classifier = separable(self.__get_frames(frame_names), **kwargs)
            self.classifiers_[group_name] = classifier
            # Merge if they are cuts of the same data set
            if are_separable:
                _logger.info(f'Found separable catalogs, merge them together')
                _logger.info(', '.join(frame_names))

                self.supersets_[group_name] = pandas.concat(self.__get_frames(frame_names))
            else:
                for n in frame_names:
                    self.supersets_[n] = self.__dataframes[n]

        return self.supersets_
