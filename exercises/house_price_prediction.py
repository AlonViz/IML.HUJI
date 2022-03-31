import sys

sys.path.insert(1, '~/IML/IML.HUJI')


from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    df = pd.read_csv(filename)

    # Add features for years since the house exists and years since it was last renovated
    df['date_formatted'] = pd.to_datetime(df['date'], format="%Y%m%dT%f", errors='coerce')
    df['yr_sold'] = pd.DatetimeIndex(df['date_formatted']).year
    df['yrs_since_built'] = df['yr_sold'] - df['yr_built']
    df['yrs_since_renovated'] = pd.DataFrame([df['yr_sold'] - df['yr_renovated'], df['yrs_since_built']]).min(axis=1)
    df['yrs_since_renovated'].fillna(df['yrs_since_built'], inplace=True)

    # Add feature: has_basement
    df['has_basement'] = df['sqft_basement']
    df['has_basement'] = df['has_basement'].where(df['has_basement'] == 0, 1)

    # Add features: average prices based on : month of sale
    df['year_month'] = pd.to_datetime(df['date_formatted']).apply(
        lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
    df['avg_price'] = df.groupby(['year_month'])['price'].transform('mean')

    # Add features: categorical feature for each zipcde
    df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)

    # Drop unneccessary features for linear regression (drop id or not?)
    df.drop(inplace=True, columns=['id', 'date', 'lat', 'long', 'date_formatted', 'yr_sold', 'year_month'])

    # Drop missing values and negatives. according to check made only a few missing values, simply drop the rows
    df.dropna(inplace=True)
    df = df[df.select_dtypes(include=[np.number]).ge(0).all(1)]

    return df.drop(columns=['price']), df['price']


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    stddev_y = np.std(y)
    for column in X.columns:
        pearson_cor = np.cov(X[column], y, bias=True)[0][1] / (stddev_y * np.std(X[column]))
        labels = {'x': column, 'y': 'Price'}
        fig = px.scatter(x=X[column], y=y, labels=labels,
                         title="Response (price) as function of feature: {column_name}<br>"
                               "<sup> Pearson correlation: {pearson}</sup>"
                         .format(column_name=column, pearson=pearson_cor))
        fig.update_layout(title_x=0.5, title_font_size=25)
        fig.write_image("{folder}/pearson_{figure_name}.png".format(folder=output_path, figure_name=column))


if __name__ == '__main__':
    # np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y, '../plots')

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    estimator = LinearRegression()
    losses = np.empty(0)
    iterations = 10

    for percent in range(1, 101):
        mean_loss = 0
        for _ in range(iterations):
            X_y_train = X_train.join(y_train)
            X_y_sample = X_y_train.sample(frac=percent / 100)
            X_sample, y_sample = X_y_sample.iloc[:, :-1], X_y_sample.iloc[:, -1:]
            estimator.fit(X_sample.values, y_sample.values)
            mean_loss += estimator.loss(X_test.values, y_test.values)
            print("percent: {percent}, iterations: {_}".format(percent=percent, _=_))

        mean_loss /= iterations
        np.append(losses, mean_loss)
    print(losses)

    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
