import numpy as np

from exercises.gradient_descent_investigation import get_gd_state_recorder_callback
from IMLearn.desent_methods.modules import L2, L1
from IMLearn.desent_methods.gradient_descent import GradientDescent

callback, values, weights = get_gd_state_recorder_callback()
L2_norm = L2(weights=np.array([np.sqrt(2), np.exp(1) / 3]))
gd = GradientDescent(callback=callback)
gd.fit(L2_norm, np.empty(0), np.empty(0))

print(values, weights)
