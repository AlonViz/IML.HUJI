from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy

output_path = "C:\\Alon\\Studies\\IML\\Exercise 4\\Graphs"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    df = pd.DataFrame(columns=['learners', 'train_error', 'test_error'])
    for partial_n_learners in range(1, n_learners + 1):
        df = df.append({'learners': partial_n_learners,
                        'train_error': adaboost.partial_loss(train_X, train_y, partial_n_learners),
                        'test_error': adaboost.partial_loss(test_X, test_y, partial_n_learners), }, ignore_index=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["learners"], y=df["train_error"], mode="lines", name="Train Error"))
    fig.add_trace(go.Scatter(x=df["learners"], y=df["test_error"], mode="lines", name="Test Error"))
    noise_text, noise_no = ('without', 'no_noise') if noise == 0 else ('with', 'noise')
    fig.update_layout(title=fr"<b>AdaBoost: Performance of learners on data "
                            fr"{noise_text} noise<b>",
                      margin=dict(t=100),
                      title_x=0.5,
                      title_font_size=20,
                      width=800,
                      height=600,
                      xaxis_title="Num. of Learners",
                      yaxis_title="Misclassification Error")
    # fig.write_image("{folder}/{figure_name}.png".format(folder=output_path,
    #                                                    figure_name=f"{noise_no}_graphs"))
    fig.show()

    symbols = np.array(['x', 'square', 'circle'])
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    losses = []
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.05, .05])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t} Learners}}$" for t in T],
                        horizontal_spacing=0.03, vertical_spacing=.05)
    for i, n_learners in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda X: adaboost.partial_predict(X, n_learners), lims[0], lims[1], density=120,
                              showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y, symbol=symbols[test_y.astype(int)],
                                    colorscale=[custom[0], custom[-1]],
                                    line=dict(color="black", width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
        losses.append(adaboost.partial_loss(test_X, test_y, n_learners))

    fig.update_layout(title=fr"<b>AdaBoost: Decision Boundaries on data {noise_text} "
                            fr"noise<br><sup> performance of models with various num. learners</sup><b>",
                      margin=dict(t=100),
                      title_x=0.5,
                      title_font_size=20,
                      width=800,
                      height=600).update_xaxes(visible=False).update_yaxes(visible=False)
    # fig.write_image("{folder}/{figure_name}.png".format(folder=output_path,
    #                                                     figure_name=f"{noise_no}_boundaries"))
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_n_learners = T[np.argmin(losses)]
    fig = go.Figure()
    fig.add_traces(
        [decision_surface(lambda X: adaboost.partial_predict(X, best_n_learners), lims[0], lims[1], density=120,
                          showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=test_y, symbol=symbols[test_y.astype(int)], colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))])

    fig.update_layout(
        title=f"<b>AdaBoost: Decision Boundary of {best_n_learners} Learners on data {noise_text} noise<br>"
              f"<sup>Accuracy on test set: {accuracy(adaboost.predict(test_X), test_y)}<b>",
        margin=dict(t=100),
        title_x=0.5,
        title_font_size=20,
        width=800,
        height=600).update_xaxes(visible=False).update_yaxes(visible=False)
    # fig.write_image("{folder}/{figure_name}.png".format(folder=output_path,
    #                                                     figure_name=f"{noise_no}_best"))
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure()
    sizes = (adaboost.D_ / np.max(adaboost.D_)) * 5
    fig.add_traces(
        [decision_surface(lambda X: adaboost.predict(X), lims[0], lims[1], density=120,
                          showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=train_y, symbol=symbols[train_y.astype(int)], colorscale=[custom[0], custom[-1]],
                                size=sizes,
                                line=dict(color="black", width=.2)))])

    fig.update_layout(title=f"<b>AdaBoost: Decision Boundary and plot of training set on data {noise_text} noise<br>"
                            f"<sup>marker size proportional to ultimate weight<b>",
                      margin=dict(t=100),
                      title_x=0.5,
                      title_font_size=20,
                      width=800,
                      height=600).update_xaxes(visible=False).update_yaxes(visible=False)
    # fig.write_image("{folder}/{figure_name}.png".format(folder=output_path,
    #                                                      figure_name=f"{noise_no}_weighted"))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=.4)
