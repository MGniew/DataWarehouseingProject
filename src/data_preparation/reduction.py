import pandas as pd
from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier


def get_reduced_dimension(data, method="pca", n_components=2):
    method = PCA(n_components=n_components)
    return pd.DataFrame(method.fit_transform(data))


def get_selected_features(data, target, method="rfe",
                          n_components=5, threshold=0.1):

    if method == "rfe":
        estimator = DecisionTreeClassifier()
        selector = RFE(estimator, n_components, step=1)
        result = selector.fit_transform(data, target)
    elif method == "vt":
        selector = VarianceThreshold()
        result = selector.fit_transform(data)
    else:
        result = SelectKBest(chi2, k=n_components).fit_transform(data, target)

    return pd.DataFrame(result)
