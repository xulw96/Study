import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib


def load_housing_data():
    return pd.read_csv("./data/housing.csv")


def main():
    # import the data
    housing = load_housing_data()
    print(housing.head())
    print(housing.info())
    print(housing["ocean_proximity"].value_counts())
    print(housing["latitude"].describe())
    '''
    # print histogram
    housing.hist(bins=50, figsize=(20, 15))  # can be replaced by plt.hist(housing["latitude"])
    plt.show()
    
    # way1 to create a test set
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    # way2 to create a test set
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]  # iloc point to number, while loc point to character.
    # way3 to create a test set
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]
    '''
    # stratified sampling
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set in (strat_train_set, strat_test_set):  # remove the income_cat attribute and back to original state
        set.drop(["income_cat"], axis=1, inplace=True)
    '''
    # data visualization
    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1) use alpha to change the showing density
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"]/100, label="population",
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True) s:radius, c:color
    plt.show()
    plt.legend()
    # correlation analysis
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    from pandas.tools.plotting import scatter_matrix
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.show()
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"] = housing["population"]/housing["households"]
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    '''
    # Data cleaning
    housing = strat_train_set.drop("median_house_value", axis=1)  # axis=1 signifies to remove the label's column
    housing_labels = strat_train_set["median_house_value"].copy()  # get new predictors and labels
    '''
    housing.dropna(subset=["total_bedrooms"])  # rid of N/A corresponding attribute
    housing.drop("total_bedrooms", axis=1)  # rid of whole attribute
    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median)  # set N/A to median value
    '''
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)  # text attribute has no median value
    imputer.fit(housing_num)  # grant median value to each attribute and stored in statistics_
    X = imputer.transform(housing_num)  # replace missing values and obtain arrays for each attribute
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)  # put the array back into DataFrame
    print(housing_tr)
    print(housing_tr.shape)

    # encode Text and Categorical Attributes into number
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    housing_cat = housing["ocean_proximity"]
    housing_cat_encoded = encoder.fit_transform(housing_cat)  # fit and transform simultaneously
    print(encoder.classes_)  # hyperparameters are accessible by public variables
    # one-hot encoding follow by
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))  # fit_transform demands 2D array
    print(housing_cat_1hot.toarray())  # from resulting 2D sparse matrix to 1D array
    # one-shot text-integer categories exchange
    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()  # adding sparse_output=True to get a sparse matrix
    housing_cat_1hot = encoder.fit_transform(housing_cat)
    print(housing_cat_1hot)  # array by default

    # custom transformers
    from sklearn.base import BaseEstimator, TransformerMixin
    rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):  # gain fit_transform from TransformerMixin
        def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room

        def fit(self, X, y=None):
            return self  # do nothing

        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
            population_per_household = X[:, population_ix] / X[:, household_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    # Transformation Pipelines
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    num_pipeline = Pipeline([('imputer', Imputer(strategy="median")),
                             ('atrribs_adder', CombinedAttributesAdder()),
                             ('std_scaler', StandardScaler()),  # another Feature Scaler is "MinMaxScaler"
                             ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    print(housing_num_tr)
    print(housing_num_tr.shape)

    # convert DataFrame into Array, which can not be handled by sklearn
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[self.attribute_names].values
    # FeatureUnion, run multiple pipelines in parallel
    from sklearn.pipeline import FeatureUnion
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    class MyLabelBinarizer(TransformerMixin):  # because LabelBinarizer support labels. Can't get into Featureunion
        def __init__(self, *args, **kwargs):
            self.encoder = LabelBinarizer(*args, **kwargs)

        def fit(self, x, y=0):
            self.encoder.fit(x)
            return self

        def transform(self, x, y=0):
            return self.encoder.transform(x)
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', MyLabelBinarizer()),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared)

    # Train model
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    print("Predictions:\t", lin_reg.predict(some_data_prepared))
    print("Labels:\t", list(some_labels))

    # evaluate RMSE
    from sklearn.metrics import mean_squared_error
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)

    # DecisionTreeRegressor
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)  # an example to warn not put test set into predictions

    # K-fold cross-validation
    from sklearn.model_selection import cross_val_score
    tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    tree_scores = np.sqrt(-tree_scores)  # the output of cross_val_score need a minus symbol before it.
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_scores = np.sqrt(-lin_scores)

    def display_scores(scores):  # for after-use also
        print("scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())
    display_scores(tree_scores)
    display_scores(lin_scores)

    # RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    forest_scores = np.sqrt(-forest_scores)
    display_scores(forest_scores)

    # Grid search
    from sklearn.model_selection import GridSearchCV
    param_grid = [
        {'n_estimators':[3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(housing_prepared, housing_labels)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)  # get best estimator directly.
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)  # get different scores for various hyperparameters
    feature_importance = grid_search.best_estimator_.feature_importances_
    print(feature_importance)  # get relative importance of each attribute
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_one_hot_attribs = list(encoder.classes_)
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    sorted(zip(feature_importance, attributes), reverse=True)
    print((feature_importance, attributes))

    # Evaluate on Test Set
    final_model = grid_search.best_estimator_
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)


if __name__ == "__main__":
    main()
