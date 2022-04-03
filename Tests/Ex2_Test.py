import numpy as np
from typing import NoReturn
from numpy.linalg import pinv
from IMLearn.utils import utils
import pandas as pd
from IMLearn.learners.regressors import LinearRegression
from IMLearn.learners.regressors import PolynomialFitting
import IMLearn.metrics.loss_functions as lossfunctions


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

y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
y_pred = np.array([199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
print(lossfunctions.mean_square_error(y_true, y_pred).__round__(3))
