import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from typing import List, Union, Tuple, Optional
numeric = Union[int, float, complex, np.number]


def low_variance(X: pd.DataFrame,
                 threshold: float = 0) -> pd.DataFrame:
    """ Feature selection based on low variance. The dataset is.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe of features.

    threshold : float
        Threshold against which the variance is calculated.
        A threshold of 0.01 means dropping the column where 99% of the values are similar.

    Example
    -------
    reduced_df = low_variance(df, 0.01)
    X_test_new_reduced = low_variance(X_test_new, 0.01)

    Returns
    -------
    Dataframe with selected features.
    """

    # Normalize the data
    normalized_df = X / X.mean()

    # Create a VarianceThreshold feature selector
    sel = VarianceThreshold(threshold=threshold)

    # Fit the selector to normalized df because higher values may have higher variances => need to adjust for that
    sel.fit(normalized_df)

    # Create a boolean mask: gives True/False value on if each feature’s Var > threshold
    mask = sel.get_support()

    # Apply the mask to create a reduced dataframe
    reduced_df = X.loc[:, mask]

    print(f"Dimensionality reduced from {X.shape[1]} to {reduced_df.shape[1]}.")
    return reduced_df
