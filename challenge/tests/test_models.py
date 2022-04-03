from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import powerpuff.agoda_cancellation_prediction as agoda_cancellation_prediction
import powerpuff.agoda_cancellation_estimator as agoda_cancellation_estimator
from sklearn.model_selection import train_test_split


def test_model_accuracy(model) -> None:
    # Load data
    df, cancellation_labels = agoda_cancellation_prediction.load_data("../train_data/agoda_cancellation_train.csv")
    train_X, test_X, train_y, test_y = train_test_split(df, cancellation_labels)
    # Fit model over data
    estimator = agoda_cancellation_estimator.AgodaCancellationEstimator().fit(train_X, train_y)
    print("Model {model} performed with accuracy :{accuracy}".format(model=model,
                                                                     accuracy=estimator.accuracy(test_X, test_y)))


if __name__ == "__main__":
    test_model_accuracy(LogisticRegression)
    test_model_accuracy(KNeighborsClassifier)
    test_model_accuracy(RandomForestClassifier)
    test_model_accuracy(DecisionTreeClassifier)
    test_model_accuracy(svm.SVC)
