import numpy as np
from typing import NoReturn
from numpy.linalg import pinv
from IMLearn.utils import utils
import pandas as pd
from IMLearn.learners.regressors import LinearRegression
from IMLearn.learners.regressors import PolynomialFitting


def _test(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
    """
    Fit Least Squares model to given samples

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data to fit an estimator for

    y : ndarray of shape (n_samples, )
        Responses of input data to fit to

    Notes
    -----
    Fits model with or without an intercept depending on value of `self.include_intercept_`
    """
    if self.include_intercept_:
        X = np.insert(X, obj=0, values=1, axis=1)
    self.coefs_ = pinv(X) @ y


X = np.array([[1, 10],
              [2, 20],
              [3, 30],
              [4, 40]])

estimator = PolynomialFitting(7)
estimator.fit(X, np.array([5, 2, 3, 4]))
print(estimator.predict(X))
print(estimator.loss(X, np.array([5, 2, 3, 4])))
