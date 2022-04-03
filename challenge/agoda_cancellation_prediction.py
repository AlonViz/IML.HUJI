import numpy as np
import pandas as pd
from powerpuff.utils.cancellation_code import evaluate_cancellation_code, no_show
from powerpuff.utils.currencies import convert_currency
from IMLearn.base import BaseEstimator
from powerpuff.agoda_cancellation_estimator import AgodaCancellationEstimator


def proccess_labels(booking_datetime: pd.Series, labels: pd.Series) -> np.ndarray:
    # print("Total: {c}".format(c=labels.size))
    cancel_duration = labels.astype('datetime64[ns]') - booking_datetime.astype('datetime64[ns]')
    # print("Cancelled: {c}".format(c=np.count_nonzero(~np.isnan(cancel_duration))))
    cancel_duration = (cancel_duration / np.timedelta64(1, 'D')).to_numpy()
    cancel_duration = np.nan_to_num(cancel_duration, nan=-1)
    cancel_duration = np.where((cancel_duration < 7) | (cancel_duration > 40) | (cancel_duration == -1), 0, 1)
    # print("Cancelled after 7 to 45 days: {0}".format(np.sum(cancel_duration)))
    return cancel_duration


def process_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features[["booking_datetime",
                         "checkin_date",
                         "checkout_date",
                         "hotel_star_rating",
                         "charge_option",
                         "guest_is_not_the_customer",
                         "no_of_adults",
                         "no_of_extra_bed",
                         "no_of_room",
                         "no_of_children",
                         "original_selling_amount",
                         "original_payment_currency",
                         "original_payment_type",
                         "is_user_logged_in",
                         "is_first_booking",
                         "cancellation_policy_code",
                         "request_nonesmoke",
                         "request_latecheckin",
                         "request_highfloor",
                         "request_largebed",
                         "request_twinbeds",
                         "request_airport",
                         "request_earlycheckin"]]

    # change categorical features to boolean
    features = pd.get_dummies(features, drop_first=True,
                              columns=['charge_option',
                                       'original_payment_type'])

    # process date features: stay_duration, booking_time_before.
    features['stay_duration'] = features['checkout_date'].astype('datetime64[ns]') \
                                - features['checkin_date'].astype('datetime64[ns]')
    features['stay_duration'] = features['stay_duration'] / np.timedelta64(1, 'D')
    features['booking_time_before'] = features['checkin_date'].astype('datetime64[ns]') \
                                      - features['booking_datetime'].astype('datetime64[ns]')
    features['booking_time_before'] = features['booking_time_before'] / np.timedelta64(1, 'D')

    features['payment_in_GBP'] = features.apply(lambda x: convert_currency(
        x['original_payment_currency'], x['original_selling_amount']), axis=1)

    features['expected_fine'] = features.apply(lambda x: evaluate_cancellation_code(x['cancellation_policy_code'],
                                                                                    x['booking_time_before'],
                                                                                    x['stay_duration']), axis=1)
    features['no_show'] = features['cancellation_policy_code'].apply(no_show)

    # drop unneccessary features
    features = features.drop(columns=["booking_datetime", "checkin_date", "checkout_date",
                                      "original_payment_currency",
                                      "cancellation_policy_code"])

    # Fill missing columns with zeros: only columns starting with 'request'...
    features = features.fillna(0)
    return features


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    labels = full_data["cancellation_datetime"]
    features = full_data.drop(columns=["cancellation_datetime"])

    labels = pd.Series(proccess_labels(features["booking_datetime"], labels))
    features = process_features(features)

    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: pd.DataFrame, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    output_df = pd.DataFrame(estimator.predict(X.to_numpy()), columns=["predicted_values"])
    output_df.to_csv(filename, index=False)


def load_test_data(columns: pd.Index, filename: str) -> pd.DataFrame:
    """
    loads test data, proccesses it in a similar manner to the train data and return it as pandas df.
    :param columns: column names of the traininst set the model was fitted on
    :param filename: csv of data to load
    """
    features = pd.read_csv(filename)
    features = process_features(features)

    # Get missing columns in the training test
    missing_cols = set(columns) - set(features.columns)
    # Add a missing column in test set with default value equal to 0
    for col in missing_cols:
        features[col] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    features = features[columns]

    return features


if __name__ == '__main__':
    np.random.seed(0)
    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(df.to_numpy(), cancellation_labels)
    # Load Test Data
    df_test = load_test_data(df.columns, "test_data/test_set_week_1.csv")
    # Store model predictions over test set
    evaluate_and_export(estimator, df_test, "results/209789924_208731968_318592052.csv")
    # ids: 209789924_208731968_318592052
