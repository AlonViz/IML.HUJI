import numpy as np
from typing import NoReturn

import sklearn.discriminant_analysis
from numpy.linalg import pinv

from IMLearn.learners.classifiers import LDA, GaussianNaiveBayes
from IMLearn.metrics import misclassification_error, accuracy
from sklearn import naive_bayes, discriminant_analysis
import plotly.express as px
from IMLearn.metrics import misclassification_error

for f in ["../datasets/gaussian1.npy", "../datasets/gaussian2.npy"]:
    # Load dataset
    dataset = np.load(f)
    X, y = dataset[:, :-1], dataset[:, -1]

    # Fit models and predict over training set
    real = sklearn.naive_bayes.GaussianNB()
    real.fit(X, y)
    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)

    print(gnb.loss(X, y))
    print(misclassification_error(real.predict(X), y))
    print()
