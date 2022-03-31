import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df['DayOfYear'] = df['Date'].apply(lambda date: date.timetuple().tm_yday)
    df = df.drop(df[df['Temp'] < -50].index)  # Drop temperature outliers
    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")
    data["YearStr"] = data["Year"].astype(str)

    # Question 2 - Exploring data for specific country
    data_israel = data[data['Country'] == 'Israel']
    years_order = list(sorted(data_israel["YearStr"].unique()))
    fig = px.scatter(data_israel, x="DayOfYear", y="Temp", color="YearStr", category_orders={"YearStr": years_order},
                     color_discrete_sequence=px.colors.sequential.Turbo)
    fig.update_layout(title="Israel: Temperatures throughout the year<br><sup> "
                            "Measured Temp. on each day of year</sup>",
                      xaxis={'title': 'Day Of Year'},
                      yaxis={'title': 'Temperature'}, title_x=0.5, title_font_size=25,
                      legend_title="Year", width=800, height=500)
    fig.show()

    df_stddev_months = data_israel.groupby("Month")["Temp"].agg(np.std)
    fig = px.bar(df_stddev_months, x=df_stddev_months.index, y="Temp")
    fig.update_layout(title="Israel: Standard Deviation of Temp. Per Month<br><sup>"
                            "Standard deviation of temperature per month, based on measurments 1995-2007</sup>",
                      xaxis={'title': 'Month', 'dtick': 1},
                      yaxis={'title': 'Standard Deviation'}, title_x=0.5, title_font_size=25,
                      legend_title="Year", width=800, height=500)
    fig.show()

    # Question 3 - Exploring differences between countries
    df_country_month = data.groupby(["Country", "Month"]).agg({"Temp": [np.mean, np.std]}).reset_index()
    df_country_month.columns = ["Country", "Month", "Mean", "Std"]

    fig = px.line(df_country_month, x="Month", y="Mean", color="Country", error_y="Std", line_shape='spline')
    fig.update_layout(title="Average Monthly Temperature",
                      xaxis={'title': 'Month', 'dtick': 1},
                      yaxis={'title': 'Temperature'}, title_x=0.5, title_font_size=25,
                      legend_title="Country", width=800, height=500)
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    X_train, y_train, X_test, y_test = split_train_test(data_israel[["DayOfYear"]], data_israel["Temp"])
    df_polynom = pd.DataFrame(columns=['Degree', 'Loss'])
    for k in range(1, 11):
        estimator = PolynomialFitting(k)
        estimator.fit(X_train.to_numpy(), y_train.to_numpy())
        df_polynom = df_polynom.append(
            {'Degree': k, 'Loss': estimator.loss(X_test.to_numpy(), y_test.to_numpy()).__round__(2)},
            ignore_index=True)

    print(df_polynom)  # Loss recorded on polynomial fitting for each value of k in [1,10]
    # Plot bar of loss for each k
    fig = px.bar(df_polynom, x="Degree", y="Loss")
    fig.update_layout(title="Polynomial Fitting: Temperature in Israel<br><sup>"
                            "MSE of polynomial model, fitted for degrees 1-10</sup>",
                      xaxis={'title': 'Polynomial Degree', 'dtick': 1},
                      yaxis={'title': 'Loss'}, title_x=0.5, title_font_size=25,
                      width=800, height=500)
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    k = 4
    estimator = PolynomialFitting(k)
    estimator.fit(data_israel[["DayOfYear"]], data_israel["Temp"])
    country_losses = pd.DataFrame(columns=["Country", "Loss"])

    Countries = ["South Africa", "The Netherlands", "Jordan"]
    for country in Countries:
        data_country = data[data["Country"] == country]
        country_losses = country_losses.append({"Country": country,
                                                "Loss": estimator.loss(data_country[["DayOfYear"]].to_numpy(),
                                                                       data_country["Temp"].to_numpy())},
                                               ignore_index=True)
    fig = px.bar(country_losses, x="Country", y="Loss")
    fig.update_layout(title="Polynomial Fitting: Performance of model fitted for israel<br><sup>"
                            "Loss on test sets of various countries</sup>",
                      title_x=0.5, title_font_size=25, width=800, height=500)
    fig.show()
