import numpy as np

class Logistic_Regression():
    
    def __init__(self, lr = 0.0001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Looping through the iterations
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Calculating gradient
            dw = (1/ n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/ n_samples) * np.sum(y_pred - y)

            #Updating values of weight and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    # Training model
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_cls

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))