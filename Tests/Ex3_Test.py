import numpy as np
from typing import NoReturn
from numpy.linalg import pinv
from IMLearn.metrics import misclassification_error, accuracy

y_true = np.concatenate([np.ones(7), np.zeros(13)])
y_pred = np.zeros(20)
# print(misclassification_error(y_true, y_pred))
# print(misclassification_error(y_true, y_pred, normalize=False))
# print(accuracy(y_true, y_pred))

classes_ = np.array([1, 2, 3])
y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
arrays = np.stack([np.where(y == class_, 1, 0) for class_ in classes_])
print(arrays)
