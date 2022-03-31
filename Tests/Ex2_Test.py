import numpy as np
from typing import NoReturn
from numpy.linalg import pinv
from IMLearn.utils import utils
import pandas as pd
from IMLearn.learners.regressors import LinearRegression


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


X = pd.DataFrame(np.array([[i for _ in range(5)] for i in range(20)]))
y = pd.DataFrame(np.array([i for i in range(20)]))
est = LinearRegression()
print(est.fit(X.values, y.values).predict(X.values))
