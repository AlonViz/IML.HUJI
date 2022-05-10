from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        n_samples, n_features = X.shape
        thresholds_errors_ = np.vstack((
            np.apply_along_axis(lambda values: (1,) + self._find_threshold(values, y, 1), axis=0, arr=X).T,
            np.apply_along_axis(lambda values: (-1,) + self._find_threshold(values, y, -1), axis=0, arr=X).T))
        # thresholds_errors is of shape (2*n_samples, 3) and contains triplets (sign, threshold, error).

        index = np.argmin(thresholds_errors_[:, -1])
        self.sign_, self.threshold_, self.j_, = thresholds_errors_[index, 0], \
                                                thresholds_errors_[index, 1], index % n_features

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        n_samples = values.size
        p = values.argsort()
        values, labels = values[p], labels[p]
        errors_ = [self._weighted_misclassification_error(np.full((labels.size,), sign), labels)]
        for i, threshold in enumerate(values[:-1]):
            errors_.append(errors_[-1] + sign * labels[i])
        threshold_index_ = np.argmin(errors_)
        return values[threshold_index_], errors_[threshold_index_] / n_samples

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self._weighted_misclassification_error(self.predict(X), y)

    @staticmethod
    def _weighted_misclassification_error(values: np.ndarray, labels: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        return sum of abs of labels, where sign(labels)!=sign(values).
        values are in (1,-1), labels don't have to be.

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against
        """
        return np.abs(labels[np.sign(labels) != np.sign(values)]).sum()

    @staticmethod
    def sign(T: int):
        """sign function. used instead of np.sign, because it might return 0s.
        returns 1 if T>=0 and --1 otherwise.
        """
        return 1 if T >= 0 else -1
