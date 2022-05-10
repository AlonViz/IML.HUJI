from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, \
    GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from IMLearn.base import BaseEstimator


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.classification_model = make_pipeline(StandardScaler(), BaggingClassifier(n_estimators=50))
        #
        # #random forest classifier model
        # for i in range(20):
        #     self.classification_model = RandomForestClassifier(n_estimators=i)
        #     #todo choose num that maximizes
        #
        # #extremly random forest classifier
        # for i in range(20):
        #     self.classification_model = ExtraTreesClassifier(n_estimators=i, max_depth=None,
        #                                                      min_samples_split=2, random_state=0)
        #
        # gradient boost
        # self.classification_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,
        #                                                        random_state=0)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to.
            responses: days to cancellation, or -1 if didn't cancel

        Notes
        -----

        """
        self.classification_model.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, 2)
            Predicted responses of given samples.
            first column is the classification prediction, second is the regression one.
        """
        # return self.classification_model.predict(X)
        return np.where(self.classification_model.predict_proba(X)[:, 1] >= 0.2, 1, 0)
        # return self.classification_model.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        ----
        """
        return self.classification_model.predict_proba(X)[:, 1]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        return f1_score(self._predict(X), y, average='macro')
