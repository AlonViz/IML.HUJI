from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    combined = np.hstack((X, y.reshape((-1, 1))))  # Combine X and y to one array of size (n_samples, n_features + 1)
    validations = np.array_split(combined, cv)  # list of K-folds.

    train_scores = []
    validation_scores = []
    for index in range(cv):
        train = np.concatenate(list((v for i, v in enumerate(validations) if i != index)))
        validation = validations[index]
        train_X, train_y, valid_X, valid_y = train[:, :-1], train[:, -1], validation[:, :-1], validation[:, -1]

        estimator.fit(train_X, train_y)
        pred_train_y, pred_valid_y = estimator.predict(train_X), estimator.predict(valid_X)
        train_scores.append(scoring(train_y, pred_train_y))
        validation_scores.append(scoring(valid_y, pred_valid_y))

    return np.average(train_scores), np.average(validation_scores)
