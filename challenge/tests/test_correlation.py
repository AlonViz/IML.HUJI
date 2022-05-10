import powerpuff.agoda_cancellation_prediction as agoda_cancellation_prediction
import numpy as np
import pandas as pd


def test_classification_correlations(df: pd.DataFrame, labels: pd.Series):
    # Print Correlations
    print("Correlations of features with labels (cancelled/not cancelled), sorted by absolute value:")
    print(df.corrwith(labels).sort_values(ascending=False, key=abs).to_string())

    # print("Covariance matrix of features:")
    # print(pd.DataFrame(np.cov(dff.to_numpy().astype(float), rowvar=False), columns=df.columns,
    #                   index=df.columns).to_string())


if __name__ == "__main__":
    dates, dff, labels = agoda_cancellation_prediction.load_data("train_data/agoda_cancellation_train.csv")
    test_classification_correlations(dff, labels)
