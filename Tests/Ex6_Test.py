import numpy as np
import pandas as pd
from IMLearn.utils.utils import split_train_test
from IMLearn.learners.classifiers import LogisticRegression
from IMLearn.desent_methods.gradient_descent import GradientDescent
from IMLearn.desent_methods.learning_rate import ExponentialLR


def test_callback(**kwargs):
	print(f"loss : {kwargs['val']}")


np.random.seed(0)
path: str = "../datasets/SAheart.data"
train_portion: float = .8
df = pd.read_csv(path)
df.famhist = (df.famhist == 'Present').astype(int)
X_train, y_train, X_test, y_test = split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)

X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy().reshape(
	-1), X_test.to_numpy(), y_test.to_numpy().reshape(-1)

solver = GradientDescent(callback=test_callback, max_iter=1000, out_type='last',
						 learning_rate=ExponentialLR(base_lr=1, decay_rate=.99))
LR = LogisticRegression(include_intercept=False, solver=solver)
LR.fit(X_train, y_train)

pred_m = LR.predict(X_train)
print(LR.loss(X_train, y_train), LR.loss(X_test, y_test))
