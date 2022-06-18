import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from tqdm import tqdm

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error

import plotly.graph_objects as go

show = False
save = True
graph_folder = "C:\\Alon\\Studies\\IML\\Exercise 6\\Graphs"


def plot_descent_path(module: Type[BaseModule],
					  descent_path: np.ndarray,
					  title: str = "",
					  xrange=(-1.5, 1.5),
					  yrange=(-1.5, 1.5)) -> go.Figure:
	"""
	Plot the descent path of the gradient descent algorithm

	Parameters:
	-----------
	module: Type[BaseModule]
		Module type for which descent path is plotted

	descent_path: np.ndarray of shape (n_iterations, 2)
		Set of locations if 2D parameter space being the regularization path

	title: str, default=""
		Setting details to add to plot title

	xrange: Tuple[float, float], default=(-1.5, 1.5)
		Plot's x-axis range

	yrange: Tuple[float, float], default=(-1.5, 1.5)
		Plot's x-axis range

	Return:
	-------
	fig: go.Figure
		Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

	Example:
	--------
	fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
	fig.show()
	"""

	def predict_(w):
		return np.array([module(weights=wi).compute_output() for wi in w])

	from utils import decision_surface
	return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
					  go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
								 marker_color="black")],
					 layout=go.Layout(xaxis=dict(range=xrange),
									  yaxis=dict(range=yrange),
									  title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
	"""
	Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

	Return:
	-------
	callback: Callable[[], None]
		Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
		at each iteration of the algorithm

	values: List[np.ndarray]
		Recorded objective values

	weights: List[np.ndarray]
		Recorded parameters
	"""
	values, weights = list(), list()

	def inner_callback(**kwargs):
		values.append(kwargs["val"])
		weights.append(kwargs["weights"])
		return

	return inner_callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
								 etas: Tuple[float] = (1, .1, .01, .001)):
	for module_class in (L1, L2):
		convergence_rates = list()
		losses = list()
		for lr in etas:
			# minimize for module, lr, initial weights:
			module = module_class(weights=init)
			callback, values, weights = get_gd_state_recorder_callback()
			gd = GradientDescent(learning_rate=FixedLR(lr), callback=callback)
			best_weights = gd.fit(module, np.empty(0), np.empty(0))
			convergence_rates.append((lr, values))
			losses.append((lr, values[-1]))

			# Plot descent path
			title = r"$\text{{Descent path of {0} norm, with fixed learning rate = {1}}}$".format(module_class.__name__,
																								  lr)
			fig = plot_descent_path(module=module_class, title=title, descent_path=np.array(weights))
			fig.update_layout(width=800,
							  height=500,
							  title_font_size=20,
							  title_x=0.5)
			if lr == .01:
				if save:
					fig.write_image(
						"{folder}/{module}_descent_path.png".format(folder=graph_folder, module=module_class.__name__))
			if show:
				fig.show()

		# Plot convergence rate
		fig2 = go.Figure([go.Scatter(x=np.arange(stop=len(values)), y=values,
									 mode="lines",
									 name=lr) for lr, values in convergence_rates])
		fig2.update_layout(title=r"$\text{{Convergence rate of {0} norm for different learning rates}}$".format(
			module_class.__name__),
			width=800,
			height=500,
			title_font_size=20,
			title_x=0.5,
			legend_title="Learning Rate",
			xaxis_title="num. iteration",
			yaxis_title="norm value")
		if show:
			fig2.show()
		if save:
			fig2.write_image(
				"{folder}/{module}_convergence_rate.png".format(folder=graph_folder, module=module_class.__name__))

		print(module_class.__name__, losses, "Fixed Rate")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
									eta: float = .1,
									gammas: Tuple[float] = (1, .99, .95, .9)):
	# Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
	for module_class in (L1, L2):
		convergence_rates = list()
		losses = list()
		for gamma in gammas:
			# minimize for module, lr, initial weights:
			module = module_class(weights=init)
			callback, values, weights = get_gd_state_recorder_callback()
			gd = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=gamma), callback=callback)
			best_weights = gd.fit(module, np.empty(0), np.empty(0))
			convergence_rates.append((gamma, values))
			losses.append((gamma, values[-1]))

			# Plot descent path for gamma=0.95
			title = r"$\text{{Descent path of {0} norm, with decay rate = {1}}}$".format(module_class.__name__, gamma)
			fig2 = plot_descent_path(module=module_class, title=title, descent_path=np.array(weights))
			fig2.update_layout(width=800,
							   height=500,
							   title_font_size=20,
							   title_x=0.5)
			if gamma == .95:
				if save:
					fig2.write_image(
						"{folder}/{module}_descent_path_exp.png".format(folder=graph_folder,
																		module=module_class.__name__))
				if show:
					fig2.show()

		# Plot algorithm's convergence for the different values of gamma
		fig = go.Figure([go.Scatter(x=np.arange(stop=len(values)), y=values,
									mode="lines",
									name=lr) for lr, values in convergence_rates])
		fig.update_layout(title=r"$\text{{Convergence rate of {0} norm for different exponential decay rates}}$".format(
			module_class.__name__),
			width=800,
			height=500,
			title_font_size=20,
			title_x=0.5,
			legend_title="Decay Rate",
			xaxis_title="num. iteration",
			yaxis_title="norm value")

		if show:
			fig.show()
		if save:
			fig.write_image(
				"{folder}/{module}_convergence_rate_exp.png".format(folder=graph_folder, module=module_class.__name__))
		print(module_class.__name__, losses, "Exponential Rate")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
		Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
	"""
	Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

	Parameters:
	-----------
	path: str, default= "../datasets/SAheart.data"
		Path to dataset

	train_portion: float, default=0.8
		Portion of dataset to use as a training set

	Return:
	-------
	train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
		Design matrix of train set

	train_y : Series of shape (ceil(train_proportion * n_samples), )
		Responses of training samples

	test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
		Design matrix of test set

	test_y : Series of shape (floor((1-train_proportion) * n_samples), )
		Responses of test samples
	"""
	df = pd.read_csv(path)
	df.famhist = (df.famhist == 'Present').astype(int)
	return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def plot_cv(values, train_errors, validation_errors):
	fig = go.Figure()
	fig.add_traces([go.Scatter(x=values, y=train_errors, mode="lines+markers",
							   line=dict(color="blue", width=2), name='Training Error'),
					go.Scatter(x=values, y=validation_errors, mode="lines+markers",
							   line=dict(color="red", width=2), name='Validation Error')])
	fig.update_layout(
		title_x=0.5,
		title_font_size=20,
		width=800,
		height=600)
	return fig


def fit_logistic_regression():
	# Load and split SA Heart Disease dataset
	X_train, y_train, X_test, y_test = load_data()
	X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy().reshape(
		-1), X_test.to_numpy(), y_test.to_numpy().reshape(-1)

	LR = LogisticRegression(solver=GradientDescent(max_iter=1000))
	LR.fit(X_train, y_train)
	y_pred_proba = LR.predict_proba(X_train)
	fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)

	fig = go.Figure([go.Scatter(x=fpr, y=tpr,
								mode="lines+markers")])
	fig.update_layout(title=r"$\text{ROC curve of logistic regression on heart disease dataset}$",
					  width=800,
					  height=500,
					  title_font_size=20,
					  title_x=0.5,
					  xaxis_title="FPR",
					  yaxis_title="TPR")
	if show:
		fig.show()
	if save:
		fig.write_image("{folder}/roc_curve.png".format(folder=graph_folder))

	best_alpha = thresholds[np.argmax(tpr - fpr)]
	LR.alpha_ = best_alpha
	best_alpha_loss = LR.loss(X_test, y_test)
	print("best alpha is {:.5f}, which achieved loss of {:.5f} on test set".format(best_alpha, best_alpha_loss))

	# Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
	for penalty in ("l1", "l2"):
		LR_REG = LogisticRegression(penalty=penalty, solver=GradientDescent(max_iter=20000))
		lamdas = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
		errors = []
		for lamda in tqdm(lamdas):
			LR_REG.lam_ = lamda
			train_error, validation_error = cross_validate(LR_REG, X_train, y_train, misclassification_error)
			errors.append((lamda, train_error, validation_error))
		errors = np.array(errors)
		if show:
			fig = plot_cv(errors[:, 0], errors[:, 1], errors[:, 2])
			fig.show()
		best_lambda = errors[np.argmin(errors[:, 2]), 0]

		LR_REG.lam_ = best_lambda
		LR_REG.fit(X_train, y_train)
		test_loss = LR_REG.loss(X_test, y_test)
		print(f"best lambda that was fitted for penalty {penalty} is {best_lambda},"
			  f" with {test_loss} miss. error on test set.")


if __name__ == '__main__':
	np.random.seed(0)
	compare_fixed_learning_rates()
	compare_exponential_decay_rates()
	fit_logistic_regression()
