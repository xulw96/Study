import tensorflow as tf

# construct and execute
x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y +2  # construct the graph

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
results = sess.run(f)
print(results)
sess.close()  # execute the graph

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()  # will call tf.get_default_session() to run in the with-block

init = tf.global_variables_initializer()  # add an init node to initialize all variables at once
with tf.Session() as sess:
    init.run()
    result = f.eval()
# Manage default_graph
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)  # set it to be default_graph locally
print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())
# Node value
w = tf.constant(3)
x = w + 2
y = x + 3
z = x * 3
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])  # run at once or the value will be evaluated twice
    print(y_val)
    print(z_val)
# Linear Regression
import numpy as np
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:  # by normal equation
    theta_value = sess.run(theta)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='x')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n+1, 1], -1, 1), name='theta')  # work like np.random.rand()
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')  # similar to np.mean()
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()  # initialize all variables in one run
with tf.Session() as sess:  # batch gradient descent
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0: # every 100 run
            print('Epoch', epoch, 'MSE =', mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
'''gradients = tf.grandients(mse, [theta])[0])  # replace the gradients computation step'''
'''optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)  # replace the two steps'''
# feed data to tf
A = tf.placeholder(tf.float32, shape=(None, 3))  # specify the shape as 3 columns, any rows
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})  # pass a feed_dict for the placeholder variables
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

'''X = tf.placeholder(tf.float32, shape=(None, n + 1), name='x')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch
with tf.Session() as sess:  # mini-batch gradient descent
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()'''
# save and restore models
'''saver = tf.compat.v1.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:  # a checkpoint at 100 runs
            save_path = saver.save(sess, 'tmp/my_model.ckpt')
        sess.run(training_op)
    best_theta = theta.eval()
    save_path = saver.save(sess, '/tmp/my_model_final.ckpt')'''
''' a with tf.Session() as sess:
    saver.restore(sess, '/tmp/my_model_final.ckpt')'''
# visualize graph
'''from datetime import datetime
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
logdir = '/tmp/run-{}/'.format(now)
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())  # save the graph and variables
with tf.Session() as sess:       
    sess.run(init)                                                                
    for epoch in range(n_epochs):    
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)  # write down the variables
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()            
file_writer.close()'''
# name scopes
with tf.name_scope('loss') as scope:  # group the variables shown the graph
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
# Modularity
def relu(X):
    with tf.name_scope('relu'):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name='weights')
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(X, w), b, name='z')
        return tf.maximum(z, 0, name='max')
'''n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name='output')  # add element-wise in lists
file_writer = tf.compat.v1.summary.FileWriter('logs/relu1', tf.get_default_graph())
file_writer.close()'''
# sharing variables
def relu(X):
    with tf.variable_scope("relu", reuse=True):  # set to reuse the variable
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="max")
X = tf.placeholder(tf.float32, shape=(None, 50), name="X")
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))  # create by calling get_variable
relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")
