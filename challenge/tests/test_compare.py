from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd


def compare_week_result(week: int):
    labels = pd.read_csv(f"labels/test_set_week_{week}_labels.csv")
    labels = labels.applymap(lambda st: int(st.split("|", )[1]))
    results = pd.read_csv(f"results/results_week_{week}.csv")
    print(classification_report(labels, results))


if __name__ == "__main__":
    compare_week_result(week=4)
