import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic import LogisticRegression


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy


da = datasets.load_breast_cancer()
X, y = da.data, da.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

if __name__ == "__main__":
        
    regressor = LogisticRegression(lr=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print("LR classification accuracy:", accuracy(y_test, predictions))