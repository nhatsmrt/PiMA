import numpy as np
from sklearn.datasets import load_iris
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# random_state = 100

iris = load_iris()
X = iris['data']
y = iris['target']

\
# lb = LabelBinarizer()
# y_enc = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)

sklearn_model = LogisticRegression(solver = 'lbfgs')
sklearn_model.fit(X_train, y_train)
predictions_sklearn = sklearn_model.predict(X_test)
print("Accuracy for Sklearn Model:")
print(np.average(np.equal(predictions_sklearn, y_test)))

class SoftmaxRegression:
    def __init__(self, n_features, n_classes):
        self._n_classes = n_classes

        self._X = tf.placeholder(shape = [None, n_features], dtype = tf.float32)
        self._W = tf.get_variable(name = "W", shape = [n_features, n_classes])
        self._b = tf.get_variable(name = "b", shape = [n_classes])

        self._z = tf.matmul(self._X, self._W) + self._b
        self._op = tf.nn.softmax(self._z)

    def fit(self, X_train, y_train, n_epochs = 10000):
        self._lb = LabelBinarizer()
        y_train_enc = self._lb.fit_transform(y_train)
        self._y = tf.placeholder(shape = [None, self._n_classes], dtype = tf.float32)
        self._loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self._y, logits = self._z)

        self._optimizer = tf.train.AdamOptimizer()
        train_step = self._optimizer.minimize(self._loss)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        for _ in range(n_epochs):
            self._sess.run(train_step, feed_dict = {self._X: X_train, self._y: y_train_enc})


    def predict(self, X_test):
        proba = self._sess.run(self._op, feed_dict = {self._X: X_test})
        return np.argmax(proba, axis = -1)

    def evaluate(self, X_test, y_test):
        y_test_enc = self._lb.transform(y_test)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self._y, axis=-1), tf.argmax(self._op, axis=-1)), tf.float32))
        return self._sess.run(accuracy, feed_dict = {self._X: X_test, self._y: y_test_enc})


tf_model = SoftmaxRegression(n_features = 4, n_classes = 3)
tf_model.fit(X_train, y_train)
print(tf_model.evaluate(X_test, y_test))
print(tf_model.evaluate(X, y))

# predictions_tf = tf_model.predict(X_test)
# print(predictions_tf)
# print(np.average(np.equal(predictions_tf, y_test)))
# print(np.average(np.equal(predictions_tf, predictions_sklearn)))
