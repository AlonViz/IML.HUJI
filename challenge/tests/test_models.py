import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IMLearn.metalearners import AdaBoost
from IMLearn.learners.classifiers import DecisionStump

import powerpuff.agoda_cancellation_prediction as agoda_cancellation_prediction


def test_classification_model(estimator, train_X, test_X, train_Y, test_Y) -> None:
    # Fit model over data
    estimator.fit(train_X, train_Y)
    print(f"Model {estimator} performed: {f1_score(estimator.predict(test_X), test_Y, average='macro')}")


def testAdaBoost():
    dates, df, cancellation_labels = agoda_cancellation_prediction.load_data("train_data/agoda_cancellation_train.csv")
    iterations = 10
    num_estimators = list(range(20, 42, 2))
    averages = []
    for n in num_estimators:
        print(n)
        scores, counts = [], []
        for _ in range(iterations):
            classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                                            n_estimators=n,
                                            learning_rate=1.0)
            train_X, test_X, train_y, test_y = train_test_split(df, cancellation_labels)
            classifier.fit(train_X, train_y)
            scores.append(f1_score(classifier.predict(test_X), test_y, average='macro'))
            counts.append(np.average(classifier.predict(test_X)))
        averages.append([n, np.average(scores), np.average(counts)])
    performance = pd.DataFrame(np.array(averages), columns=['n_estimators', 'score', 'guess-rate'])
    print(performance)


def testBagging():
    dates, df, cancellation_labels = agoda_cancellation_prediction.load_data("train_data/agoda_cancellation_train.csv")
    num_estimators = list(range(20, 42, 2))
    for n in num_estimators:
        classifier = BaggingClassifier(n_estimators=n)
        train_X, test_X, train_y, test_y = train_test_split(df, cancellation_labels)
        classifier.fit(train_X, train_y)
        print(n, f1_score(classifier.predict(test_X), test_y, average='macro'))


def testEx4AdaBoost():
    dates, df, cancellation_labels = agoda_cancellation_prediction.load_data("train_data/agoda_cancellation_train.csv")
    cancellation_labels = np.where(cancellation_labels == 0, -1, 1)
    iterations = 10
    num_estimators = list(range(20, 42, 2))
    averages = []
    for n in num_estimators:
        print(n)
        scores, counts = [], []
        for _ in range(iterations):
            classifier = AdaBoost(DecisionStump, n)
            train_X, test_X, train_y, test_y = train_test_split(df, cancellation_labels)
            classifier.fit(train_X, train_y)
            scores.append(f1_score(classifier.predict(test_X), test_y, average='macro'))
            counts.append(np.average(classifier.predict(test_X)))
        averages.append([n, np.average(scores), np.average(counts)])
    performance = pd.DataFrame(np.array(averages), columns=['n_estimators', 'score', 'guess-rate'])
    print(performance)


if __name__ == "__main__":
    testBagging()
    testEx4AdaBoost()
