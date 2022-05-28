from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from powerpuff.agoda_cancellation_estimator import AgodaCancellationEstimator
from powerpuff.agoda_cancellation_prediction import load_data
from powerpuff.agoda_cancellation_prediction import load_test_data


def test_evaluate_and_export(estimator, X: pd.DataFrame, filename: str):
    labels = pd.DataFrame(estimator.predict(X.to_numpy()), columns=["predicted_values"])
    labels.to_csv(filename, index=False)


def evaluate_over_train(df, cancellation_labels):
    """
    returns f1-macro score of estimator over SAME train set.
    :return:
    """
    estimator = AgodaCancellationEstimator()
    estimator.fit(df, cancellation_labels)
    return f1_score(estimator.predict(df), cancellation_labels, average='macro')


def evaluate_over_split(df, cancellation_labels, test_size: Union[int, float] = .25):
    """
    returns f1-macro score of estimator over splitted train set.
    :param test_size: size of test set in split, either int (num of samples) or float (fraction of set)
    :return:
    """
    estimator = AgodaCancellationEstimator()
    train_X, test_X, train_Y, test_Y = train_test_split(df, cancellation_labels, test_size=test_size)
    estimator.fit(train_X, train_Y)

    return classification_report(estimator.predict(test_X), test_Y)


def evaluate_over_test(estimator, df, cancellation_labels, week: int = 1):
    """
    returns f1-macro score of estimator over test set of given week.
    :param week: week to evaluate f1-macro on.
    """


    # Load Test data
    dates_test, df_test = load_test_data(df.columns, f"test_data/test_set_week_{week}.csv")
    labels = pd.read_csv(f"labels/week_{week}_labels.csv")
    labels = labels.applymap(lambda st: int(st.split("|", )[1]))
    return classification_report(estimator.predict(df_test.to_numpy()), labels)


def evaluate_optimal_threshold(df, cancellation_labels, test_size: Union[int, float] = .25):
    """
    find threshoold that maximizes f1-score
    :param df:
    :param cancellation_labels:
    :param test_size:
    :return:
    """
    estimator = AgodaCancellationEstimator()
    train_X, test_X, train_Y, test_Y = train_test_split(df, cancellation_labels, test_size=test_size)
    estimator.fit(train_X, train_Y)

    probabilities = estimator.predict_proba(test_X)
    thresholds = np.sort(np.unique(probabilities))
    df = pd.DataFrame(thresholds, columns=['threshold'])
    f1s = []
    for threshold in thresholds:
        f1s.append(f1_score(np.where(probabilities >= threshold, 1, 0), test_Y, average='macro'))
    df['f1s'] = f1s
    # print(df)
    return classification_report(estimator.predict(test_X), test_Y)


if __name__ == "__main__":
    dates, df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    estimator = AgodaCancellationEstimator()
    estimator.fit(df.to_numpy(), cancellation_labels.to_numpy())
    for week in range(1, 5):
        print(f"TEST: {evaluate_over_test(estimator, df, cancellation_labels, week=week)}")
    for _ in range(1000):
        # print(f"{evaluate_optimal_threshold(df, cancellation_labels)}")
        print(f"TRAIN: {evaluate_over_split(df, cancellation_labels)}")
