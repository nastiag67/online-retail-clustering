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

    print(f"Dimensionality reduced from {X.shape[1]} to {reduced_df.shape[1]-1}.")
    return reduced_df


def RFE_selection(X: pd.DataFrame,
                  y: pd.Series,
                  n_features_to_select: int,
                  step: int,
                  mask: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Recursive Feature Elimination based on random forest classifier.

    Parameters
    ----------
    n_features_to_select : int
        Number of features to be selected.
    step : int
        How many features to remove at each step.
    mask : default=None
         Existing feature selection filter, which can be used to select features on testing dataset.


    Example
    -------
    n_features_to_select=300
    reduced_df, mask = RFE_selection(df, n_features_to_select=n_features_to_select, step=1, mask=None)
    X_test_new_reduced = RFE_selection(X_test_new, n_features_to_select=n_features_to_select, step=1, mask=mask)

    Returns
    -------
    If mask is None:
        reduced_df : pd.DataFrame
            Dataframe with selected features.
    If mask is not None:
        reduced_df : pd.DataFrame
            Dataframe with selected features.
        mask :
            Feature selection filter.
    """
    df = pd.concat([X, y], axis=1)

    if mask is not None:
        # Apply the mask to the feature dataset X
        reduced_df = df.loc[:, mask]
        return reduced_df

    # DROP THE LEAST IMPORTANT FEATURES ONE BY ONE
    # Set the feature eliminator to remove 2 features on each step
    rfe = RFE(estimator=RandomForestClassifier(random_state=42),
              n_features_to_select=n_features_to_select,
              step=step,
              verbose=0)

    # Fit the model to the training data
    rfe.fit(X, y)

    # Create a mask: remaining column names
    mask = rfe.support_

    # Apply the mask to the feature dataset X
    reduced_X = X.loc[:, mask]
    reduced_df = pd.concat([reduced_X, y], axis=1)

    print(f"Dimensionality reduced from {df.shape[1]} to {reduced_df.shape[1]-1}.")
    return reduced_df, mask
