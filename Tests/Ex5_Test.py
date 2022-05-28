import numpy as np
import pandas as pd
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.learners.classifiers.perceptron import Perceptron
from IMLearn.learners.classifiers.decision_stump import DecisionStump
from IMLearn.learners.classifiers.gaussian_naive_bayes import GaussianNaiveBayes
from IMLearn.metrics import accuracy, mean_square_error

df = pd.read_csv("datasets/SAheart.data", header=0, index_col=0)
df.famhist = df.famhist == "Present"
X, y = df.loc[:, df.columns != 'chd'].values, df["chd"].values
y = np.where(y == 0, -1, 1)
estimator = DecisionStump()
cv = 4
cv = cross_validate(estimator, X.astype(int), y.astype(int), accuracy, cv)
print(cv)
