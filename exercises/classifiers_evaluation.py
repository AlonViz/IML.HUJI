import numpy as np
import pandas as pd
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple

from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    def perceptron_callback(fit: Perceptron, x: np.ndarray, res: int):
        """
        callback function to be given to fitted perceptron.
        fit is the Perceptron object being fitted, x the sample the perceptron was wrong on, y response.
        """
        losses.append(fit.loss(X, y))

    for n, f in [("Linearly Separable", "../datasets/linearly_separable.npy"),
                 ("Linearly Inseparable", "../datasets/linearly_inseparable.npy")]:
        # Load dataset
        dataset = np.load(f)
        X, y = dataset[:, :-1], dataset[:, -1]

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=perceptron_callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(x=np.arange(1, len(losses) + 1, 1), y=losses)
        fig.update_layout(title_text=f"Fitting Perceptron With {n} Data:<br><sup>"
                                     "Misclassification error during algorithm iterations</sup>",
                          xaxis_title="Iteration",
                          yaxis_title="Loss", title_x=0.5,
                          title_font_size=25,
                          height=500,
                          width=800)
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker=dict(color="black"))


def get_marker(mu: np.ndarray):
    """
    Draw a marker centered at given location.

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of marker
    """
    return go.Scatter(x=[mu[0]], y=[mu[1]], mode="markers", marker=dict(color="black",
                                                                        size=10,
                                                                        symbol="x"))


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for n, f in [("Gaussian-1", "../datasets/gaussian1.npy"), ("Gaussian-2", "../datasets/gaussian2.npy")]:
        models = [("Naive Bayes", GaussianNaiveBayes), ("LDA", LDA)]
        model_names = [model[0] for model in models]
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{m}$" for m in model_names],
                            horizontal_spacing=0.05, vertical_spacing=.03)
        for i, (name, model) in enumerate(models):
            # Load dataset
            # Fit models and predict over training set
            dataset = np.load(f)
            X, y = dataset[:, :-1], dataset[:, -1]
            classifier = model()
            classifier.fit(X, y)
            classes = classifier.predict(X)

            df = pd.DataFrame(np.column_stack((X, y, classes)),
                              columns=["Feature 1", "Feature 2", "class", "prediction"])

            fig.add_trace(go.Scatter(x=df["Feature 1"], y=df["Feature 2"], mode="markers", showlegend=False,
                                     marker=dict(color=df["prediction"], symbol=df["class"],
                                                 colorscale=custom[0:3],
                                                 line=dict(color="black", width=1))),
                          col=(i + 1), row=1)
            fig.layout.annotations[i].update(text=f"{name}, accuracy: {accuracy(y, classes).__round__(3)}")

            for j, class_ in enumerate(classifier.classes_):
                if type(classifier) is GaussianNaiveBayes:
                    fig.add_trace(get_ellipse(classifier.mu_[j, :], np.diag(classifier.vars_[j, :])), col=(i + 1),
                                  row=1)
                elif type(classifier) is LDA:
                    fig.add_trace(get_ellipse(classifier.mu_[j, :], classifier.cov_), col=(i + 1), row=1)
                fig.add_trace(get_marker(classifier.mu_[j, :]), col=(i + 1), row=1)

            # Add traces for data-points setting symbols and colors
            # raise NotImplementedError()
            # Add `X` dots specifying fitted Gaussians' means
            # raise NotImplementedError()
            # Add ellipses depicting the covariances of the fitted Gaussians
            # raise NotImplementedError()

        fig.update_layout(title=fr"<b>Classification: Performance of probabilistic classifiers on {n} data<b>",
                          margin=dict(t=100),
                          title_x=0.5,
                          title_font_size=25,
                          width=1000,
                          height=600,
                          showlegend=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
