import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from plotly.subplots import make_subplots

import IMLearn.metrics
from IMLearn.learners.classifiers import LDA, GaussianNaiveBayes
from IMLearn.metrics import accuracy
from exercises.classifiers_evaluation import get_ellipse, get_marker
from utils import custom
from sklearn.naive_bayes import GaussianNB

m = 1000

vars = [0.5, 1, 0.5, 1]


def gaussians_non_linear():
    mu = [np.array([5, 2]), np.array([3, -1]), np.array([-4, 2])]

    x, y = [], []
    for _m in range(m):
        y.append(int(np.random.choice([0, 1, 2])))
        x.append(mu[y[-1]] + np.random.multivariate_normal([0, 0], [[4, 0.5], [0.5, 2]]))

    y = np.array(y)
    return np.array(x), y


if __name__ == "__main__":
    X, y = gaussians_non_linear()
    gnb_m, gnb_r = LDA(), LinearDiscriminantAnalysis(store_covariance=True)

    gnb_m.fit(X, y)
    gnb_r.fit(X, y)

    # print(accuracy(gnb_r.predict(X), y))
    # print(accuracy(gnb_m.predict(X), y))
    # print()
    # print(gnb_m.pi_)
    # print(gnb_r.priors_)
    # print()
    # print(gnb_m.mu_)
    # print(gnb_r.means_)
    #
    # print(gnb_m.cov_)
    # print(gnb_r.covariance_)

    classes = gnb_m.predict(X)
    fig = go.Figure()
    i=0
    name = "LDA"
    classifier = gnb_m

    df = pd.DataFrame(np.column_stack((X, y, classes)),
                      columns=["Feature 1", "Feature 2", "class", "prediction"])

    fig.add_trace(go.Scatter(x=df["Feature 1"], y=df["Feature 2"], mode="markers", showlegend=False,
                             marker=dict(color=df["prediction"], symbol=df["class"],
                                         colorscale=custom[0:3],
                                         line=dict(color="black", width=1))))
    print(accuracy(y,classes))

    for j, class_ in enumerate(classifier.classes_):
        if type(classifier) is GaussianNaiveBayes:
            fig.add_trace(get_ellipse(classifier.mu_[j, :], np.diag(classifier.vars_[j, :])))
        elif type(classifier) is LDA:
            fig.add_trace(get_ellipse(classifier.mu_[j, :], classifier.cov_))
        fig.add_trace(get_marker(classifier.mu_[j, :]))

    # Add traces for data-points setting symbols and colors
    # raise NotImplementedError()
    # Add `X` dots specifying fitted Gaussians' means
    # raise NotImplementedError()
    # Add ellipses depicting the covariances of the fitted Gaussians
    # raise NotImplementedError()

    fig.update_layout(title=fr"<b>Classification: Performance of probabilistic classifiers on data<b>",
                      margin=dict(t=100),
                      title_x=0.5,
                      title_font_size=25,
                      width=1000,
                      height=600,
                      showlegend=False)
    fig.show()


