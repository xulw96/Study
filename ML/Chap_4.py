import numpy as np
import matplotlib.pyplot as plt

def main():
    # normal equation for Linear Regression
    x = 2 * np.random.rand(100, 1)  # it returns an array of 100 rows; 1 column
    y = 4 + 3 * x + np.random.randn(100, 1)  # distribute as "normal random numbers" instead of "equal random numbers"

    """plt.scatter(x, y)
    plt.show()"""

    x_b = np.c_[np.ones((100, 1)), x]  # add x0=1 to each instance
    theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)  # using attribute to do calculation
    print("theta_best:", theta_best)

    x_new = np.array([[0], [2]])
    x_new_b = np.c_[np.ones((2, 1)), x_new]
    y_predict = x_new_b.dot(theta_best)
    print("predicted value:", y_predict)

    """plt.plot(x_new, y_predict, "r-")
    plt.plot(x, y, "b.")  # using "b." attribute to avoid using scatter plot
    plt.axis([0, 2, 0, 15])
    plt.show()"""

    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()  # do same thing with Sklearn
    lin_reg.fit(x, y)
    print("lin_reg.intercept is", lin_reg.intercept_)
    print("lin_reg.coef is", lin_reg.coef_)

    # Batch Gradient Descent
    eta = 0.1
    n_iterations = 1000
    m = 100
    theta = np.random.randn(2, 1)  # random initialization
    for iteration in range(n_iterations):
        gradients = 2 / m * x_b.T.dot(x_b.dot(theta) - y)  # the gradient vector of the cost function
        theta = theta - eta * gradients
    print("final theta value:", theta)

    """def plot_gradient_descent(theta, eta, theta_path=None):
        m = len(x_b)
        plt.plot(x, y, "b.")
        n_iterations = 1000
        for iteration in range(n_iterations):
            if iteration < 10:
                y_predict = x_new_b.dot(theta)
                style = "b-" if iteration > 0 else "r--"  # note the expression here
                plt.plot(x_new, y_predict, style)
            gradients = 2 / m * x_b.T.dot(x_b.dot(theta) - y)
            theta = theta - eta * gradients
        plt.xlabel("$x_1$", fontsize=18)
        plt.axis([0, 2, 0, 15])
        plt.title(r"$\eta = {}$".format(eta), fontsize=16)"""

    """np.random.seed(42)
    theta = np.random.randn(2, 1)  # initialization
    plt.figure(figsize=(10, 4))
    plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(132); plot_gradient_descent(theta, eta=0.1)
    plt.subplot(133); plot_gradient_descent(theta, eta=0.5)
    plt.show()"""
    # Stochastic gradient descent
    """m = len(x_b)
    np.random.seed(42)
    n_epochs = 50
    t0, t1 = 5, 50  # learning schedule hyperparameters
    def learning_schedule(t):
        return t0 / (t + t1)
    theta = np.random.randn(2, 1)
    for epoch in range(n_epochs):
        for i in range(m):
            if epoch == 0 and i < 20:
                y_predict = x_new_b.dot(theta)
                style = "b-" if i >0 else "r--"
                plt.plot(x_new, y_predict, style)
            random_index = np.random.randint(m)
            xi = x_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients  # with a changing learning rate
    plt.plot(x, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.show()
    print("theta is:", theta)"""

    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)  # run 50 epochs, starting learning rate is 0.1
    sgd_reg.fit(x, y.ravel())  # ravel() shape the array into one column
    print("sgd_reg.intercept is:", sgd_reg.intercept_)
    print("sgd_reg.coef is:", sgd_reg.coef_)

    # Polynominal regression
    m = 100  # m stand for training instances; n stand for features
    x = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)  # generate nonlinear data
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree=2, include_bias=False)  # transform to add square features
    x_poly = poly_features.fit_transform(x)
    print("original x is:", x[0])
    print("x, after extending is:", x_poly[0])
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    print("lin_ewf.intercept is:", lin_reg.intercept_)
    print("lin_reg.coef is:", lin_reg.coef_)

    # Learning curves
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    def plot_learning_curves(model, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        train_errors, val_errors = [], []
        for m in range(1, len(x_train)):
            model.fit(x_train[:m], y_train[:m])
            y_train_predict = model.predict(x_train[:m])
            y_val_predict = model.predict(x_val)  # for validation, use all data
            train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict, y_val))
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
        plt.legend(loc="upper right", fontsize=14)
        plt.xlabel("Training set size", fontsize=14)
        plt.ylabel("RMSE", fontsize=14)
    """lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, x, y)
    plt.show()"""

    # Ridge regression
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    np.random.seed(42)
    m = 20
    x = 3 * np.random.rand(m, 1)
    y = 1 + 0.5 * x + np.random.randn(m, 1) / 1.5
    x_new = np.linspace(0, 3, 100).reshape(100, 1)  # grant 100 numbers at the interval between 0 and 3

    """def plot_model(model_class, polynomial, alphas, **model_kargs):
        for alpha, style in zip(alphas, ("b-", "g--", "r:")):
            model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
            if polynomial:
                model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])
            model.fit(x, y)
            y_new_regul = model.predict(x_new)
            lw = 2 if alpha > 0 else 1
            plt.plot(x_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
        plt.plot(x, y, "b.", linewidth=3)
        plt.legend(loc="upper left", fontsize=15)
        plt.xlabel("$x_1$", fontsize=18)
        plt.axis([0, 3, 0, 4])
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
    plt.show()"""

    from sklearn.linear_model import Ridge
    ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)  # using cholesky matrix factorization
    ridge_reg.fit(x, y)
    print("predicted value from ridge regression is:", ridge_reg.predict([[1.5]]))

    sgd_reg = SGDRegressor(penalty="l2")  # the penalty attribute grant the "l2" norm of weight vector
    sgd_reg.fit(x, y.ravel())  # grant 1d array instead of column-vector
    print("predicted value from SGDRegressior (l2 penalty) is:", sgd_reg.predict([[1.5]]))

    # Lasso regression
    from sklearn.linear_model import Lasso
    """plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1, random_state=42)
    plt.show()"""

    from sklearn.linear_model import Lasso
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(x, y)
    print("predicted value from Lasso regresion is:", lasso_reg.predict([[1.5]]))

    sgd_reg = SGDRegressor(penalty="l1")
    sgd_reg.fit(x, y.ravel())
    print("predicted value from SGDRegressor (l1 penalty) is:", sgd_reg.predict([[1.5]]))

    # Elastic net
    from sklearn.linear_model import ElasticNet
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)  # l1_ratio is the mix ratio
    elastic_net.fit(x, y)
    print("predicted value from Elasticnet is:", elastic_net.predict([[1.5]]))

    # Early stopping
    np.random.seed(42)
    m = 100
    x = 6 * np.random.rand(m, 1) - 3
    y = 2 + x + 0.5 * x**2 + np.random.rand(m, 1)
    x_train, x_val, y_train, y_val = train_test_split(x[:50], y[:50].ravel(), test_size=0.5, random_state=10)
    poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler())
    ])
    x_train_poly_scaled = poly_scaler.fit_transform(x_train)
    x_val_poly_scaled = poly_scaler.fit_transform(x_val)  # provide the dataset

    sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, penalty=None, eta0=0.0005,
                           warm_start=True, learning_rate="constant", random_state=42)
    n_epochs = 500
    train_errors, val_errors = [], []
    for epoch in range(n_epochs):
        sgd_reg.fit(x_train_poly_scaled, y_train)
        y_train_predict = sgd_reg.predict(x_train_poly_scaled)
        y_val_predict = sgd_reg.predict(x_val_poly_scaled)
        train_errors.append(mean_squared_error(y_train, y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    """best_epoch = np.argmin(val_errors)
    best_val_rmse = np.sqrt(val_errors[best_epoch])
    plt.annotate('Best model', xy=(best_epoch, best_val_rmse),
                 xytext=(best_epoch, best_val_rmse + 1),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=16,)
    best_val_rmse = 0.03  # mannly set to make the graph better
    plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.show()"""

    from sklearn.base import clone
    sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                           penalty=None, learning_rate="constant",
                           eta0=0.0005, random_state=42)
    minimum_val_error = float("inf")
    best_epoch = None
    best_model = None
    for epoch in range(1000):
        sgd_reg.fit(x_train_poly_scaled, y_train)  # warm_start enable it to continue the fit
        y_val_predict = sgd_reg.predict(x_val_poly_scaled)
        val_error = mean_squared_error(y_val, y_val_predict)
        if val_error < minimum_val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)
    print("best_epoch is:", best_epoch)
    print("best_model is:", best_model)

    # logistic regression
    from sklearn import datasets
    iris = datasets.load_iris()
    x = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # y=1 for iris-virginica.

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver="liblinear", random_state=42)
    log_reg.fit(x, y)
    print("Logistic model info:", log_reg)
    x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(x_new)
    decision_boundary = x_new[y_proba[:, 1] >= 0.5][0]  # need more knowledge on numpy to understand this
    print("decision boundary value is:", decision_boundary)

    """plt.plot(x_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
    plt.plot(x_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("petal width", fontsize=14)
    plt.ylabel("probability", fontsize=14)
    plt.show()"""

    """from sklearn.linear_model import LogisticRegression
    x = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.int)
    log_reg = LogisticRegression(solver="liblinear", C=10**10, random_state=42)
    log_reg.fit(x, y)
    x0, x1 = np.meshgrid(np.linspace(2.0, 7, 500).reshape(-1, 1),
                         np.linspace(0.8, 2.7, 200).reshape(-1, 1))
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_proba = log_reg.predict_proba(x_new)
    plt.figure(figsize=(10, 4))
    plt.plot(x[y == 0, 0], x[y == 0, 1], "bs")
    plt.plot(x[y == 1, 0], x[y == 1, 1], "g^")
    zz = y_proba[:, 1].reshape(x0.shape)
    contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
    left_right = np.array([2.9, 7])
    boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0] / log_reg.coef_[0][1])
    plt.clabel(contour, inline=1, fontsize=12)
    plt.plot(left_right, boundary, "k--", linewidth=3)
    plt.text(3.5, 1.5, "Not Iris_Virginica", fontsize=14, color="b", ha="center")
    plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.axis([2.9, 7, 0.8, 2.7])
    plt.show()"""

    # Softmax regression
    from sklearn import datasets
    iris = datasets.load_iris()
    x = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
    softmax_reg.fit(x, y)

    """x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                         np.linspace(0, 3.5, 200).reshape(-1, 1))
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_proba = softmax_reg.predict_proba(x_new)
    y_predict = softmax_reg.predict(x_new)
    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)
    plt.figure(figsize=(10, 4))
    plt.plot(x[y == 2, 0], x[y == 2, 1], "g^", label="Iris-Virginica")
    plt.plot(x[y == 1, 0], x[y == 1, 1], "bs", label="Iris-Versicolor")
    plt.plot(x[y == 0, 0], x[y == 0, 1], "yo", label="Iris-Setosa")
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
    plt.clabel(contour, inline=1, fontsize=12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 7, 0, 3.5])
    plt.show()"""


if __name__ == "__main__":
    main()
