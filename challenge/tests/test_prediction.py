from sklearn.model_selection import train_test_split
import powerpuff.agoda_cancellation_prediction as agoda_cancellation_prediction
import powerpuff.agoda_cancellation_estimator as agoda_cancellation_estimator


def test_accuracy():
    # Load data
    df, cancellation_labels = agoda_cancellation_prediction.load_data("../train_data/agoda_cancellation_train.csv")
    train_X, test_X, train_y, test_y = train_test_split(df, cancellation_labels)
    # Fit model over data
    estimator = agoda_cancellation_estimator.AgodaCancellationEstimator().fit(train_X, train_y)
    print("Accuracy:{accuracy}".format(accuracy=estimator.accuracy(test_X, test_y)))


def test_correlations():
    # Load data
    dff, labels = agoda_cancellation_prediction.load_data("../train_data/agoda_cancellation_train.csv")
    # Print Correlations
    print("Correlations:")
    print(dff.corrwith(labels).sort_values(ascending=False, key=abs).to_string())


if __name__ == "__main__":
    test_correlations()
    test_accuracy()
