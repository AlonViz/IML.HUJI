from scipy.stats import norm
from IMLearn.learners.gaussian_estimators import UnivariateGaussian
import numpy as np

samples_ = np.random.normal(0, 2, 10)
lgpdf_true = norm.logpdf(samples_, loc=0, scale=2)
calcpdf = lambda x: UnivariateGaussian.log_likelihood(0, 4, x)
calcpdfvec = np.vectorize(calcpdf)
lgpdf_mine = calcpdfvec(samples_)
lgpdf_mine = np.around(lgpdf_mine, 2)
lgpdf_true = np.around(lgpdf_true, 2)
assert (np.array_equal(lgpdf_mine, lgpdf_true))
