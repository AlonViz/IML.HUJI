import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IMLearn.learners.classifiers import LDA, GaussianNaiveBayes
from IMLearn.metrics import accuracy
from utils import custom


def gaussians_non_linear():
    mu = [np.array([-5, -5]), np.array([-5, 5]), np.array([5, -5]), np.array([5, 5])]

    x, y = [], []
    for _m in range(m):
        y.append(int(np.random.choice([0, 1, 2, 3])))
        x.append(mu[y[-1]] + np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]]))

    y = np.array(y)
    return (np.array(x), np.logical_or(y == 1, y == 2).astype(int)), "Four Gaussians Two Classes"
