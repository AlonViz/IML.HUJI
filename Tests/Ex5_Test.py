import numpy as np
from IMLearn.learners.regressors.linear_regression import LinearRegression
from IMLearn.learners.regressors.ridge_regression import RidgeRegression

# Generate X,y
X = np.random.multivariate_normal(mean=[0, 0, 0], cov=np.diag([1, 3, 2]), size=100)
w = np.array([1, 5, -20])
b = 123
y = X @ w + b + np.random.normal(loc=0, scale=5, size=100)

# compare performances of Linear and Ridge. for lam=0, should be the same.
# for lam > 0, ridge regression might have higher loss, but coefs. values should be smaller.
lr = LinearRegression(include_intercept=True).fit(X, y)
rr = RidgeRegression(lam=1, include_intercept=True).fit(X, y)
print(lr.coefs_, rr.coefs_)
print(lr.loss(X, y), rr.loss(X, y))
