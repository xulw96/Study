import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# Perceptron
iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int)
per_clf = Perceptron(max_iter=100, tol=-np.infty, random_state=618)
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])
# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def relu(z):
    return np.maximum(0, z)
def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps)) / (2 * eps)
def heaviside(z):
    return (z >= 0).astype(z.dtype)
def mlp_xor(x1, x2, activation=heaviside):
    return activation(-activation(x1 + x2 - 1.5) + activation(x1 + x2 - 0.5) -0.5)
# Estimator API
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_train.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
def high_level_ANN():
    feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
    dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_cols)
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)
    dnn_clf.train(input_fn=input_fn)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_test}, y=y_test, shuffle=False)
    eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
    print(eval_results)
# plain Tensorflow
init = tf.global_variables_initializer()
saver = tf.train.Saver()
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):  # split into multiple subarrays
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
def ann_implementation():
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int32, shape=(None), name='y')
    def neuron_layer(X, n_neurons, name, activation=None):  # can be substituted by tf functions
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name='kernel')  # weight matrix
            b = tf.Variable(tf.zeros([n_neurons]), name='bias')
            Z = tf.matmul(X, W) + b
            if activation is not None:
                return activation(Z)
            else:
                return Z
    with tf.name_scope('dnn'):
        hidden1 = neuron_layer(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
        logits = neuron_layer(hidden2, n_outputs, name='outputs')  # output not going through activation
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
    learning_rate = 0.01
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)  # passing integer labels; without converting to one-hot
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # cast the checking result to be number, not boolearn value
    n_epochs = 40
    batch_size = 50
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, 'Batch accuracy:', acc_batch, 'Val accuracy:', acc_val)
        save_path = saver.save(sess, './ckpt/my_model_final.ckpt')

    with tf.Session() as sess:
        '''saver.restore(sess, './ckpt/my_model_final.ckpt')
        X_new_scaled = X_test[:20]
        Z = logits.eval(feed_dict={X: X_new_scaled})
        y_pred = np.argmax(Z, axis=1)'''  # restore from ckpt
    print('predicted classes:', y_pred)
    print('actual classes:', y_test[:20])
# tensorflow api, dense()
def dense_ann():
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int32, shape=(None), name='y')
    with tf.name_scope('dnn'):
        hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
        logits = tf.layers.dense(hidden2, n_outputs, name='outputs')
        y_proba = tf.nn.softmax(logits)  # result measured by softmax
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
    learning_rate = 0.01
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    n_epochs = 20
    n_batches = 50
    batch_size = 50
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y:y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, 'Batch accuracy', acc_batch, 'Validation accuracy:', acc_valid)
        save_path = saver.save(sess, './ckpt/my_model_final.ckpt')
