from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from IMLearn.metrics import mean_square_error
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge as RidgeSKLearn
from sklearn.model_selection import train_test_split
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plot = False


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
	"""
	Simulate data from a polynomial model and use cross-validation to select the best fitting degree

	Parameters
	----------
	n_samples: int, default=100
		Number of samples to generate

	noise: float, default = 5
		Noise level to simulate in responses
	"""
	# Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
	# and split into training- and testing portions
	f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

	X = np.linspace(-1.2, 2.0, n_samples)
	y_true = (np.vectorize(f)(X))
	train_X, test_X, train_y, test_y = train_test_split(X, y_true, train_size=2 / 3)
	train_y_noised = train_y + np.random.normal(0, noise, train_y.size)
	test_y_noised = test_y + np.random.normal(0, noise, test_y.size)

	fig = go.Figure()
	fig.add_traces([go.Scatter(x=train_X, y=train_y, mode="markers",
							   marker=dict(color='blue', symbol=class_symbols[0],
										   line=dict(color="black", width=1)), name='Train set'),
					go.Scatter(x=test_X, y=test_y, mode="markers",
							   marker=dict(color='red', symbol=class_symbols[0],
										   line=dict(color="black", width=1)), name='Test set')])
	fig.update_layout(title=fr"<b>Noiseless data split into train and test sets."
							fr"<br><sup> Sampled uniformly from y = (x + 3)(x + 2)(x + 1)(x - 1)(x - 2)</sup><b>",
					  title_x=0.5,
					  title_font_size=20,
					  width=800,
					  height=600,
					  xaxis_title="x",
					  yaxis_title="f(x)")

	if noise == 5 and plot:
		fig.show()

	# Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
	train_X = train_X.reshape(-1, 1)
	training_errors = []
	validation_errors = []
	cv = 5
	degrees = list(range(0, 11))
	for k in degrees:
		estimator = PolynomialFitting(k=k)
		train_error, validation_error = cross_validate(estimator, train_X, train_y_noised, mean_square_error, cv=cv)
		training_errors.append(train_error)
		validation_errors.append(validation_error)
	fig = go.Figure()
	fig.add_traces([go.Scatter(x=degrees, y=training_errors, mode="lines+markers",
							   line=dict(color="blue", width=2), name='Training Error'),
					go.Scatter(x=degrees, y=validation_errors, mode="lines+markers",
							   line=dict(color="red", width=2), name='Validation Error')])
	fig.update_layout(
		title=fr"<b>MSE on train set and validation set with 5-Fold CV<br><sup>{n_samples} Samples, Noise: {noise}</sup>",
		title_x=0.5,
		title_font_size=20,
		width=800,
		height=600,
		xaxis_title="Polynomial Degree",
		yaxis_title="Mean Squared Error"
	)
	if plot:
		fig.show()

	# Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
	k_star = int(np.argmin(validation_errors))
	estimator = PolynomialFitting(k=k_star)
	estimator.fit(train_X, train_y_noised)
	test_error = mean_square_error(estimator.predict(test_X), test_y_noised).__round__(2)
	print(f"Best Results: noise={noise},  fitted degree k={k_star}, "
		  f"validation error={np.min(validation_errors).__round__(2)} with test error: {test_error}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
	"""
	Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
	values for Ridge and Lasso regressions

	Parameters
	----------
	n_samples: int, default=50
		Number of samples to generate

	n_evaluations: int, default = 500
		Number of regularization parameter values to evaluate for each of the algorithms
	"""
	# Question 6 - Load diabetes dataset and split into training and testing portions
	data, target = load_diabetes(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=n_samples)

	# Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
	learners = [("Lasso", Lasso, 'alpha'), ("Ridge", RidgeRegression, 'lam'), ("RidgeSKLearn", RidgeSKLearn, 'alpha')]
	learners_best_params = []
	reg_values = np.linspace(0.02, 1, n_evaluations)
	for name, learner, param_name in learners:
		training_errors = []
		validation_errors = []
		for i, reg_term in enumerate(reg_values):
			estimator = learner(**{param_name: reg_term})
			train_error, validation_error = cross_validate(estimator, X_train, y_train, mean_square_error, 5)
			training_errors.append(train_error)
			validation_errors.append(validation_error)

		learners_best_params.append(reg_values[np.argmin(validation_errors)])

		fig = go.Figure()
		fig.add_traces([go.Scatter(x=reg_values, y=training_errors, mode="lines",
								   line=dict(color="blue", width=2), name='Training Error'),
						go.Scatter(x=reg_values, y=validation_errors, mode="lines",
								   line=dict(color="red", width=2), name='Validation Error')])
		fig.update_layout(
			title=fr"<b>{name} model losses, as a function of regularization param.</sup>",
			title_x=0.5,
			title_font_size=20,
			width=800,
			height=600,
			xaxis_title="Regularization term",
			yaxis_title="MSE"
		)
		if plot:
			fig.show()

	# Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
	for i, learner in enumerate(learners):
		print(f"the best param for {learner[0]} is {learners_best_params[i]}")

	learners.append(("Linear Regression", LinearRegression, None))

	for i, (name, learner, param) in enumerate(learners):
		estimator = learner(**{param: learners_best_params[i]}) if param is not None else learner()
		estimator.fit(X_train, y_train)
		error = mean_square_error(estimator.predict(X_test), y_test)
		print(f"estimator {name} achieved {error} error on test set")


if __name__ == '__main__':
	np.random.seed(0)
	# select_polynomial_degree(100, 5)
	# select_polynomial_degree(100, 0)
	# select_polynomial_degree(1500, 10)
	select_regularization_parameter(50, 500)
