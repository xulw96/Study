import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():
    # Linear SVM
    from sklearn import datasets
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    iris = datasets.load_iris()
    x = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge"))  # remember to add loss as "hinge"
    ])
    svm_clf.fit(x, y)
    print("predicted value for [5.5, 1.7] is:", svm_clf.predict([[5.5, 1.7]]))

    # Nonlinear SVM
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import PolynomialFeatures
    x, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ])
    polynomial_svm_clf.fit(x, y)

    def plot_dataset(x, y, axes):
        plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], "bs")
        plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], "g^")
        plt.axis(axes)
        plt.grid(True, which='both')
        plt.xlabel(r"$x_1$", fontsize=20)
        plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    def plot_predictions(clf, axes):
        x0s = np.linspace(axes[0], axes[1], 100)
        x1s = np.linspace(axes[2], axes[3], 100)
        x0, x1 = np.meshgrid(x0s, x1s)
        x = np.c_[x0.ravel(), x1.ravel()]
        y_pred = clf.predict(x).reshape(x0.shape)
        y_decision = clf.decision_function(x).reshape(x0.shape)  # it grants the decision boundary
        plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
        plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
    """plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
    plt.show()"""

    # Kernel trick
    from sklearn.svm import SVC
    polynomial_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
    polynomial_kernel_svm_clf.fit(x, y)
    polynomial100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
    polynomial100_kernel_svm_clf.fit(x, y)

    """plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plot_predictions(polynomial_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r"$d=3, r=1, C=5$", fontsize=18)
    plt.subplot(122)
    plot_predictions(polynomial100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r"$d=10, r=100, C=5$", fontsize=18)
    plt.show()"""

    # RBF kernel
    from sklearn.svm import SVC
    gamma1, gamma2 = 0.1, 5
    C1, C2 = 0.001, 1000
    hyperparams = (gamma1, C1), (gamma2, C2), (gamma2, C1), (gamma2, C2)
    svm_clfs = []
    for gamma, C in hyperparams:
        rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
        rbf_kernel_svm_clf.fit(x, y)
        svm_clfs.append(rbf_kernel_svm_clf)
    plt.figure(figsize=(11, 7))
    for i, svm_clf in enumerate(svm_clfs):
        plt.subplot(221 + i)
        plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
        plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
        gamma, C = hyperparams[i]
        plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
    """plt.show()"""

    # SVR
    np.random.seed(42)
    m = 100
    x = 2 * np.random.rand(m, 1) - 1
    y = (0.2 + 0.1 * x + 0.5 * x**2 + np.random.randn(m, 1) / 10).ravel()
    from sklearn.svm import SVR, LinearSVR
    svm_reg = LinearSVR(epsilon=0.1)
    svm_reg.fit(x, y)
    svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="auto")  # epsilon-sensitive: epsilon controls the street width
    svm_poly_reg2 = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1, gamma="auto")
    svm_poly_reg1.fit(x, y)
    svm_poly_reg2.fit(x, y)
    def plot_svm_regression(svm_reg, x, y, axes):
        xls = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
        y_pred = svm_reg.predict(xls)
        plt.plot(xls, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
        plt.plot(xls, y_pred + svm_reg.epsilon, "k--")
        plt.plot(xls, y_pred - svm_reg.epsilon, "k--")
        plt.scatter(x[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
        plt.plot(x, y, "bo")
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.legend(loc="upper left", fontsize=18)
        plt.axis(axes)
    """plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plot_svm_regression(svm_poly_reg1, x, y, [-1, 1, 0, 1])
    plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg1.degree, svm_poly_reg1.C, svm_poly_reg1.epsilon), fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.subplot(122)
    plot_svm_regression(svm_poly_reg2, x, y, [-1, 1, 0, 1])
    plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg2.degree, svm_poly_reg2.C, svm_poly_reg2.epsilon), fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.show()"""

if __name__ == "__main__":
    main()
