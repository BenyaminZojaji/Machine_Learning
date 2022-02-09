import numpy as np
import pandas as pd

class LinearLeastSquare:
    def __init__(self):
        pass
    # train
    def fit(self, X, Y):
        # w = slope
        # w = (X.T X)^-1 * X.T Y
        self.w = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
        return self.w

    def predict(self, x):
        height_pred = x * self.w
        return height_pred

    def evaluate(self, X, Y, loss='MAE'):
        Y_pred = np.matmul(X, self.w)
        Error = np.abs(Y - Y_pred)

        if loss=='MAE':
            MAE = np.mean(Error)
            return MAE
        elif loss == 'MSE':
            MSE = np.mean(Error**2)
            return MSE
        elif loss == 'Huber':
            is_small_error = np.abs(Error) < 1
            squared_loss = np.square(Error) / 2
            linear_loss  = np.abs(Error) - 0.5
            Huber = np.where(is_small_error, squared_loss, linear_loss)
            return Huber
        elif loss == 'Hinge':
            new_predicted = np.array([-1 if i==0 else i for i in Y_pred])
            new_actual = np.array([-1 if i==0 else i for i in Y])
            Hinge = np.mean([max(0, 1-x*y) for x, y in zip(new_actual, new_predicted)])
            return Hinge