# from sklearn.model_selection import train_test_split
# from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from SVM import Multiclass_SVM

# X, y = datasets.make_blobs(
#     n_samples=50, n_features=2, centers= 2, cluster_std=1.05, random_state=40
# )

import numpy as np

X_train = np.array([
    [1, 2], [2, 3], [3, 4],   # class 0
    [5, 6], [6, 7], [7, 8],   # class 1
    [9, 10], [10, 11], [11, 12], # class 2
])

y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

X_test = np.array([
    [2, 3],   # should predict 0
    [6, 7],   # should predict 1
    [10, 11]  # should predict 2
])
y_test = np.array([0, 1, 2])

# y = np.where(y == 0, -1, 1)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=123
# )

clf = Multiclass_SVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/ len(y_true)
    return accuracy

print("SVM classification accuracy", accuracy(y_test, predictions))
print("Predictions:", predictions)
print("Expected:   ", y_test)