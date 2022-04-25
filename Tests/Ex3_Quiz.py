from IMLearn.learners.classifiers import GaussianNaiveBayes, LDA
import numpy as np
import pandas as pd

models = [("Naive Bayes", GaussianNaiveBayes)]
for i, (name, model) in enumerate(models):
    # Load dataset
    # Fit models and predict over training set
    dataset = np.array([[0, 0], [1, 0], [2, 1], [3, 1], [4, 1], [5, 1], [6, 2], [7, 2]])
    X, y = dataset[:, :-1], dataset[:, -1]
    classifier = model()
    classifier.fit(X, y)
    classes = classifier.predict(X)

    print("expectation")
    print(classifier.mu_)
    print("prior")
    print(classifier.pi_)
    print()

models = [("Naive Bayes", GaussianNaiveBayes)]
for i, (name, model) in enumerate(models):
    # Load dataset
    # Fit models and predict over training set
    dataset = np.array([[1, 1, 0], [1, 2, 0], [2, 3, 1], [2, 4, 1], [3, 3, 1], [3, 4, 1]])
    X, y = dataset[:, :-1], dataset[:, -1]
    classifier = model()
    classifier.fit(X, y)
    classes = classifier.predict(X)

    print("variance")
    print(classifier.vars_)
