import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.discriminant_analysis
from plotly.subplots import make_subplots

import IMLearn.metrics
from IMLearn.learners.classifiers import LDA, GaussianNaiveBayes
from IMLearn.metrics import accuracy
from utils import custom

m = 1000

vars = [0.5, 1, 0.5, 1]


def gaussians_non_linear():
    mu = [np.array([-1, 1]), np.array([1, -1]), np.array([-1, -1]), np.array([1, 1])]

    x, y = [], []
    for _m in range(m):
        y.append(int(np.random.choice([0, 1, 2, 3])))
        x.append(mu[y[-1]] + np.random.multivariate_normal([0, 0], [[1, 1], [1, 1]]))

    y = np.array(y)
    return np.array(x), y


if __name__ == "__main__":
    X, y = gaussians_non_linear()
    lda, gnb = LDA(), GaussianNaiveBayes()
    lda2 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(store_covariance=True)
    lda.fit(X, y)
    lda2.fit(X, y)
    gnb.fit(X, y)

    df = pd.DataFrame(np.column_stack((X, y, lda.predict(X))),
                      columns=["Feature 1", "Feature 2", "class", "prediction"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Feature 1"], y=df["Feature 2"], mode="markers", showlegend=False,
                             marker=dict(color=df["prediction"], symbol=df["class"],
                                         colorscale=custom[0:3],
                                         line=dict(color="black", width=1))))
    # fig.show()
