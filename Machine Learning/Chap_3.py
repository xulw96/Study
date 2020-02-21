import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def main():

    # import the data
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784')
    x, y = mnist["data"], mnist["target"]
    print(x.shape)
    print(y.shape)

    # show the image
    some_digit = x[36000]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

    # prepare the testing/training tests
    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
    np.random.seed(3)
    shuffle_index = np.random.permutation(60000)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

    # Binary Classifier
    y_train_5 = (y_train == '5')  # True for all 5s
    y_test_5 = (y_test == '5')  # make sure it's int not chars
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(x_train, y_train_5)  # enable the model
    print(sgd_clf.predict([some_digit]))

    # implement Cross-Validation
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(x_train, y_train_5):
        clone_clf = clone(sgd_clf)  # train clone on training folds, then predict on test fold
        x_train_folds = x_train[train_index]
        y_train_folds = y_train_5[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train_5[test_index]
        clone_clf.fit(x_train_folds, y_train_folds)
        y_pred = clone_clf.predict(x_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))

    # evaluate the model with 'accuracy'
    from sklearn.model_selection import cross_val_score
    cross_val_score = cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy")
    print(cross_val_score)

    # see accuracy from a non5classifier
    from sklearn.base import BaseEstimator
    class Never5Classifier(BaseEstimator):
        def fit(self, x, y=None):
            pass
        def predicit(self, x):
            return np.zeros((len(x), 1), dtype=bool)
    never_5_clf = Never5Classifier()
    never_5_clf_score = cross_val_score(never_5_clf, x_train, y_train_5, cv=3, scoring="accuracy")
    print(never_5_clf_score)

    # evaluate the model with 'confusion matrix'
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)
    confusion_matrix = confusion_matrix(y_train_5, y_train_pred)
    print(confusion_matrix)

    # precision and recall
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision_score = precision_score(y_train_5, y_train_pred)
    recall_score = recall_score(y_train_5, y_train_pred)
    f1_score = f1_score(y_train_5, y_train_pred)
    print(precision_score)
    print(recall_score)
    print(f1_score)  # f1 score is the harmonic mean of precision and recall

    # precision vs recall trade-off
    from sklearn.metrics import precision_recall_curve

    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="precision")  # function to plot precision vs threshold
        plt.plot(thresholds, recalls[:-1], "g-", label="recall")
        plt.xlabel("Threshold", fontsize=16)
        plt.legend(loc="upper left", fontsize=16)
        plt.ylim([0, 1])

    def plot_precision_vs_recall(precisions, recalls):
        plt.plot(recalls, precisions, "b-", linewidth=2)
        plt.xlabel("recall", fontsize=16)
        plt.ylabel("precision", fontsize=16)
        plt.axis([0, 1, 0, 1])
    y_scores = cross_val_predict(sgd_clf, x_train, y_train, cv=3, method="decision_function")  # return decision value
    if y_scores.ndim == 2:
        y_scores = y_scores[:, 1]  # to get around with the issue of "extra first dimension"
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plot_precision_vs_recall(precisions, recalls)
    plt.show()

    # manly set the threshold
    y_train_pred_90 = (y_scores > 70000)  # gain new trained dataset
    precision_score = precision_score(y_train_5, y_train_pred_90)
    recall_score = recall_score(y_train_5, y_train_pred_90)
    print("precision_score=", precision_score)
    print("recall_score=", recall_score)

    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    plot_roc_curve(fpr, tpr)
    plt.show()
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3, method="predict_proba")  # have no decision_function
    y_scores_forest = y_probas_forest[:, 1]  # extract the score from probability metrics
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
    plt.plot(fpr, tpr, "b:", label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "random Forest")
    plt.legend(loc="lower right")
    plt.show()
    roc_auc_score = roc_auc_score(y_train_5, y_scores_forest)
    print(roc_auc_score)

    # Multiclass classification
    sgd_clf.fit(x_train, y_train)  # train the model to the all set.
    sgd_clf.predict([some_digit])
    some_digit_score = sgd_clf.decision_function([some_digit])  # obtain score for each class
    print(some_digit_score)

    # OvO classifier
    from sklearn.multiclass import OneVsOneClassifier
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
    ovo_clf.fit(x_train, y_train)
    print(ovo_clf.predict([some_digit]))
    forest_clf.fit(x_train, y_train)
    print(forest_clf.predict_proba([some_digit]))
    sgd_clf_score = cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy")
    print(sgd_clf_score)  # here the score is for multiclass classification as for y_train
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
    sgd_clf_score(sgd_clf, x_train_scaled, y_train, cv=3, scoring="accuracy")
    print(sgd_clf_score)  # scaling can improve the accuracy for model

    # error analysis
    y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)  # row for actual, column for predicted
    print(conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)  # showing the matrix with a image
    plt.show()
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums  # transform error number into error rate
    np.fill_diagonal(norm_conf_mx, 0)  # keep only the errors
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()

    # multilabel classification
    from sklearn.neighbors import KNeighborsClassifier
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)  # imply odd number in this way
    y_multilabel = np.c_[y_train_large, y_train_odd]
    knn_clf = KNeighborsClassifier()  # KNeighborClassifier for multilabel
    knn_clf.fit(x_train, y_multilabel)
    print(knn_clf.predcit([some_digit]))

    # multioutput classification
    import numpy.random as rnd
    noise1 = rnd.randint(0, 100, len(x_train), 784)
    noise2 = rnd.randint(0, 100, (len(x_train), 784))  # grant noise and try to clean
    x_train_mod = x_train +noise1
    x_test_mod = x_test + noise2
    y_train_mod = x_train
    y_test_mod = x_test
    knn_clf.fit(x_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([x_test_mod[1]])
    plot_digit(clean_digit)


if __name__ == "__main__":
    main()
