from typing import Tuple
import numpy as np
import pandas as pd
from powerpuff.utils.cancellation_code import evaluate_cancellation_code, no_show, fine_after_x_days
from powerpuff.utils.currencies import convert_currency
from IMLearn.base import BaseEstimator
from powerpuff.agoda_cancellation_estimator import AgodaCancellationEstimator
import pycountry
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None  # default='warn'


def process_labels(booking_datetime: pd.Series, day_of_month: pd.Series, labels: pd.Series) -> np.ndarray:
    """
    process the labels. gets booking_datetime (date) and labels (date) and returns [0,1] labels:
    1 iff guest has canceled in the 7-13th of the next month.
    :param booking_datetime column of data
    :param day_of_month column of data
    :param labels column of data
    :return:
    """

    def has_cancelled(booking_mon, label_month, label_day):
        return label_month == ((booking_mon + 1) % 12) and 7 <= label_day <= 13

    def has_cancelled_at_all(booking_mon, label_month, label_day):
        return label_month != 0 or label_day != 0

    # Different method of processing labels.
    # cancel_labels = labels.astype('datetime64[ns]') - booking_datetime.astype('datetime64[ns]')
    # cancel_labels = (cancel_labels / np.timedelta64(1, 'D')).to_numpy()
    # cancel_labels = np.nan_to_num(cancel_labels, nan=-1)
    # cancel_labels = np.ceil(cancel_labels).astype(int)
    # cancel_labels += day_of_month
    # cancel_labels = np.vectorize(lambda x: 1 if 37 <= x <= 43 else 0)(cancel_labels)

    booking_month = pd.to_datetime(booking_datetime).apply(lambda date: date.timetuple().tm_mon)
    labels_month = pd.to_datetime(labels).apply(lambda date: 0 if pd.isnull(date) else date.timetuple().tm_mon)
    labels_day = pd.to_datetime(labels).apply(lambda date: 0 if pd.isnull(date) else date.timetuple().tm_mday)
    cancel_labels = np.vectorize(has_cancelled)(booking_month, labels_month, labels_day)

    # Next line: labels - cancelled/ not cancelled
    # cancel_labels = np.vectorize(lambda date: 0 if pd.isnull(date) else 1)(labels)
    return cancel_labels


def process_features(features: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Main pre-processing function of the features. all preprocessing should be done here.
    :param features:
    :return:
    """
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
                         "original_payment_method",
                         "is_user_logged_in",
                         "is_first_booking",
                         "cancellation_policy_code",
                         "request_nonesmoke",
                         "request_latecheckin",
                         "request_highfloor",
                         "request_largebed",
                         "request_twinbeds",
                         "request_airport",
                         "request_earlycheckin",
                         "customer_nationality",
                         "accommadation_type_name",
                         "hotel_country_code",
                         "hotel_chain_code",
                         "language"]]

    def is_same_country(country_name, country_code):
        if pd.isnull(country_code) or pd.isnull(country_name):
            return False
        return pycountry.countries.get(alpha_2=country_code).name == country_name

    features['same_country'] = features.apply(lambda x: is_same_country(x['customer_nationality'],
                                                                        x['hotel_country_code']), axis=1)

    features['day_of_month'] = features['booking_datetime'].apply(lambda date: date.timetuple().tm_mday)
    features['month'] = features['booking_datetime'].apply(lambda date: date.timetuple().tm_mon)

    features['month_has_31_days'] = features['month'].apply(lambda x: x in [1, 3, 5, 7, 8, 10, 12])

    # process date features: stay_duration, booking_time_before.
    features['stay_duration'] = features['checkout_date'].astype('datetime64[ns]') \
                                - features['checkin_date'].astype('datetime64[ns]')
    features['stay_duration'] = features['stay_duration'] / np.timedelta64(1, 'D')
    features['booking_time_before'] = features['checkin_date'].astype('datetime64[ns]') \
                                      - features['booking_datetime'].astype('datetime64[ns]')
    features['booking_time_before'] = features['booking_time_before'] / np.timedelta64(1, 'D')

    features['is_one_person'] = (features['no_of_adults'] == 1)

    features['payment_in_GBP'] = features.apply(lambda x: convert_currency(
        x['original_payment_currency'], x['original_selling_amount']), axis=1)

    features['no_show'] = features['cancellation_policy_code'].apply(no_show)

    features['checkin_day_of_week'] = features['checkin_date'].astype('datetime64[ns]').apply(
        lambda date: date.weekday())

    features['hotel_chain_code'] = np.where(pd.isnull(features['hotel_chain_code']), 0, 1)

    features['is_english'] = features['language'].apply(lambda x: "English" in x)

    features['has_children'] = np.where(features['no_of_children'] > 0, 1, 0)
    features['vacation_importance'] = features['booking_time_before'] * features['stay_duration']
    features['expected_fine'] = features.apply(lambda x: evaluate_cancellation_code(x['cancellation_policy_code'],
                                                                                    x['booking_time_before'],
                                                                                    x['stay_duration']), axis=1)
    features["most_popular_cancellation"] = np.where(features["cancellation_policy_code"] == "365D100P_100P", 1, 0)

    features["fine_after_one_day"] = features.apply(lambda x: fine_after_x_days(x['cancellation_policy_code'],
                                                                                x['booking_time_before'],
                                                                                x['stay_duration'], 1), axis=1)

    features["fine_after_ten_days"] = features.apply(lambda x: fine_after_x_days(x['cancellation_policy_code'],
                                                                                 x['booking_time_before'],
                                                                                 x['stay_duration'], 10), axis=1)

    features["fine_after_thirty_days"] = features.apply(lambda x: fine_after_x_days(x['cancellation_policy_code'],
                                                                                    x['booking_time_before'],
                                                                                    x['stay_duration'], 30), axis=1)
    # drop unnecessary features
    features = features.drop(columns=["checkin_date", "checkout_date",
                                      "original_payment_currency",
                                      "cancellation_policy_code",
                                      "language", "hotel_country_code", "customer_nationality"])

    # change categorical features to boolean
    features = pd.get_dummies(features, drop_first=True,
                              columns=['charge_option',
                                       'original_payment_type',
                                       'checkin_day_of_week',
                                       'original_payment_method',
                                       'accommadation_type_name'])

    # Fill missing columns with zeros: only columns starting with 'request'...
    features = features.fillna(0)
    booking_datetime = features["booking_datetime"]
    features = features.drop(columns=["booking_datetime"])
    return booking_datetime, features


def load_data(filename: str) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
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
    full_data = pd.read_csv(filename, parse_dates=['booking_datetime', 'cancellation_datetime']).drop_duplicates()
    labels = full_data["cancellation_datetime"]
    features = full_data.drop(columns=["cancellation_datetime"])

    booking_datetime, features = process_features(features)
    labels = pd.Series(process_labels(booking_datetime, features["day_of_month"], labels))
    return booking_datetime, features, labels


def evaluate_and_export(AgodaEstimator: BaseEstimator, X: pd.DataFrame, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    AgodaEstimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    labels = pd.DataFrame(AgodaEstimator.predict(X.to_numpy()), columns=["predicted_values"])
    labels.to_csv(filename, index=False)


def load_test_data(columns: pd.Index, filename: str) -> Tuple[pd.Series, pd.DataFrame]:
    """
    loads test data, proccesses it in a similar manner to the train data and return it as pandas df.
    :param columns: column names of the traininst set the model was fitted on
    :param filename: csv of data to load
    """
    features = pd.read_csv(filename, parse_dates=['booking_datetime'])
    booking_datetime, features = process_features(features)

    # Get missing columns in the training test
    missing_cols = set(columns) - set(features.columns)
    # Add a missing column in test set with default value equal to 0
    for col in missing_cols:
        features[col] = 0
    features = features.copy()
    # Ensure the order of column in the test set is in the same order as in train set
    features = features[columns]

    return booking_datetime, features


if __name__ == '__main__':
    np.random.seed(0)
    # Load data
    dates, df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(df.to_numpy(), cancellation_labels.to_numpy())
    # Load Test Data
    weeks = [5]
    for week in weeks:
        dates_test, df_test = load_test_data(df.columns, f"test_data/test_set_week_{week}.csv")
        # Store model predictions over test set
        evaluate_and_export(estimator, df_test, f"results/results_week_{week}.csv")
        # ids: 209789924_208731968_318592052
