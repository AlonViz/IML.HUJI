from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.io as pio
import plotly.express as px
import pandas as pd

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    """Create a sample of size 1000 of N(10,1) and print the expectaton and
    variance estimations of the univariate gaussian estimator"""
    mu_, stdev_ = 10, 1
    var_ = stdev_ * stdev_
    samples_ = np.random.normal(loc=mu_, scale=stdev_, size=1000)
    estimator = UnivariateGaussian()
    estimator.fit(samples_)
    print(estimator.mu_, estimator.var_)

    # Question 2 - Empirically showing sample mean is consistent
    """using samples of increasing sizes(10,20,...,1000), plot the difference
     of the estimated and real expectation as a function of sample size."""
    sample_estimator = UnivariateGaussian()
    estimations = list()
    for size in range(10, 1010, 10):
        sample_estimator.fit(samples_[:size])
        estimations.append(np.abs(mu_ - sample_estimator.mu_))
    # Plot:
    df_sample_size = pd.DataFrame(
        np.array([list(range(10, 1010, 10)), estimations]).transpose(),
        columns=["Sample Size", "Estimation Error"])
    fig_1 = px.scatter(df_sample_size, x="Sample Size", y="Estimation Error")
    fig_1.update_layout(
        title_text='Univariate Gaussian Estimator<br><sup> Error in'
                   ' expectancy as a function of sample size</sup>'
        , title_x=0.5, title_font_size=25)
    fig_1.show()

    # Question 3: Plot the PDF using fitted model
    """Compute the PDF of the previously drawn samples using the model fitted 
    in question 1. Plot the empirical PDF function under the fitted model"""
    # Create:
    samples_.sort()
    pdfs_ = estimator.pdf(samples_)
    fig_2 = px.scatter(x=samples_, y=pdfs_)
    fig_2.update_layout(title_text="Univariate Gaussian Estimator<br><sup>"
                                   " sample density plotted on empirical PDF</sup>",
                        xaxis_title="Sample Value",
                        yaxis_title="Empirical PDF", title_x=0.5,
                        title_font_size=25)
    fig_2.update_traces(marker=dict(size=2))
    fig_2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    """Fit a multivariate Gaussian and print the estimated expectation
     and covariance matrix."""
    mu_ = np.array([0, 0, 4, 0])
    cov_ = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0],
                     [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples_ = np.random.multivariate_normal(mu_, cov_, size=1000)
    estimator = MultivariateGaussian()
    estimator.fit(samples_)
    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    """Using the samples drawn in the question above calculate the log-likelihood
    for models with expectation µ = [f1,0,f3,0]. Plot a heatmap of f1 values
     as rows, f3 values as columns and the color being the calculated log likelihood."""
    sample_count = 200  # needs to be 200
    f1 = np.linspace(-10, 10, sample_count)
    f3 = np.linspace(-10, 10, sample_count)
    func = lambda x, y: MultivariateGaussian.log_likelihood(
        np.array([x, 0, y, 0]), cov_, samples_)
    func_vec = np.vectorize(func)
    res = func_vec(f1[:, np.newaxis], f3)

    labels_dict = {"x": "f3", "y": "f1", "color": "Log-likelihood"}
    fig = px.imshow(res, x=f1, y=f3, labels=labels_dict)
    fig.update_layout(title_text="Multivariate Gaussian Estimator<br><sup>"
                                 "Log-likelihood of expectation µ = [f1,0,f3,0] and known covariance, "
                                 "values drawn with expectation [0,0,4,0]</sup>",
                      title_x=0.5, title_font_size=25,
                      legend_x=0)
    fig.show()

    # Question 6 - Maximum likelihood
    """Of all values tested in question 5, which model (pair of values for 
    feature 1 and 3) achieved the maximum log-likelihood value? Round to 3 decimal places"""
    argmax_tup = np.unravel_index(res.argmax(), res.shape)
    print("Max value achieved: {0}".format(res.max()))
    print("Argmax: f1 = {0}, f3 = {1}".format(f1[argmax_tup[0]],
                                              f3[argmax_tup[1]]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
