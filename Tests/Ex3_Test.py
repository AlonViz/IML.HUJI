import numpy as np
from typing import NoReturn
from numpy.linalg import pinv
import IMLearn.metrics.loss_functions as lossfunctions

y_true = np.concatenate([np.ones(13), np.zeros(7)])
y_pred = np.zeros(20)
print(lossfunctions.misclassification_error(y_true, y_pred))
print(lossfunctions.misclassification_error(y_true, y_pred, normalize=False))
