import numpy as np
from typing import NoReturn

import sklearn.discriminant_analysis
from numpy.linalg import pinv

from IMLearn.learners.classifiers import LDA
from IMLearn.metrics import misclassification_error, accuracy
from sklearn import naive_bayes, discriminant_analysis
import plotly.express as px
from IMLearn.metrics import misclassification_error

for f in ["../datasets/gaussian1.npy", "../datasets/gaussian2.npy"]:
    # Load dataset
    dataset = np.load(f)
    X, y = dataset[:, :-1], dataset[:, -1]

    # Fit models and predict over training set
    real = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(store_covariance=True)
    real.fit(X, y)
    lda = LDA()
    lda.fit(X, y)

    print(real.priors)
    print(lda.pi_)
    print()
    print(real.means_)
    print(lda.mu_)
    print()
    print(real.classes_)
    print(lda.classes_)
    print()
    print(real.covariance_)
    print(lda.cov_)
    print()
