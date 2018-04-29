import numpy as np
from sklearn.datasets import load_boston
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 32

boston = load_boston()
y = boston['target'].reshape(-1, 1).astype(np.float32)
X = boston['data']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = RANDOM_STATE)


class LinearRegressor:
    def __init__(self, n_features):

        self._X = tf.placeholder(shape = [None, n_features], dtype = tf.float32)
        self._W = tf.get_variable(name = "W", shape = [n_features, 1])
        self._b = tf.get_variable(name = "b", shape = [1])

        self._op = tf.matmul(self._X, self._W) + self._b

    def fit(self, X_train, y_train, n_epochs = 100000):
        self._y = tf.placeholder(shape = [None, 1], dtype = tf.float32)
        self._loss = tf.reduce_mean(tf.square(self._y - self._op))

        self._optimizer = tf.train.AdamOptimizer()
        train_step = self._optimizer.minimize(self._loss)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        for _ in range(n_epochs):
            self._sess.run(train_step, feed_dict = {self._X: X_train, self._y: y_train})


    def predict(self, X_test):
        return self._sess.run(self._op, feed_dict = {self._X: X_test})



tf_model = LinearRegressor(n_features = 13)
tf_model.fit(X_train, y_train)
predictions_tf = tf_model.predict(X_test)
print("Error from Tensorflow model:")
print(mean_squared_error(predictions_tf, y_test))


sklearn_model = LinearRegression(fit_intercept = True)
sklearn_model.fit(X_train, y_train)
predictions_sklearn = sklearn_model.predict(X_test)
print("Error from Sklearn model:")
print(mean_squared_error(predictions_sklearn, y_test))



# def linear_regression(X, y):
#     return np.matmul(np.matmul(inv(np.matmul(X.T, X)), X.T), y)


# W = linear_regression(X_train, y_train)
# print("Error from numpy model:")
# print(mean_squared_error(np.matmul(X_test, W), y_test))


