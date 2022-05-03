import numpy as np
from IMLearn.learners.classifiers.decision_stump import DecisionStump

np.set_printoptions(threshold=np.inf, linewidth=200)

stump = DecisionStump()

values = np.array([range(0, 10*k, k) for k in range(1, 11)]).T
labels = np.array([-1, -1, -1, -1, -1, -1, -1, 1, -1, 1])

stump.fit(values, labels)
print(stump.predict(values))
